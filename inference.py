# -*- coding: utf-8 -*-
"""
Licensing Information: 
Please do not distribute or publish solutions to this project.
You are free to use and extend Driverless Car for educational purposes.
The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.

This module implements two tracking algorithms:
1. ExactInference: Maintains belief distribution via full grid traversal.
2. ParticleFilter: Approximates belief distribution using a set of particles.
"""

import collections
import math
import random
import util
from engine.const import Const
from util import Belief


class ExactInference(object):
    """
    Maintains and updates a belief distribution over the probability of a car
    being in a tile using exact updates (correct, but slow at scale).
    """

    def __init__(self, numRows: int, numCols: int):
        self.skipElapse = False
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()

    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        """
        Updates the belief distribution based on a distance observation.
        Uses Gaussian PDF for emission probability and normalizes the belief.
        """
        for row in range(self.belief.getNumRows()):
            for col in range(self.belief.getNumCols()):
                old_prob = self.belief.getProb(row, col)
                if old_prob == 0:
                    continue

                car_x = util.colToX(col)
                car_y = util.rowToY(row)

                dist = math.sqrt((agentX - car_x) ** 2 + (agentY - car_y) ** 2)
                
                # Calculate emission probability (Gaussian distribution)
                emission_prob = util.pdf(dist, Const.SONAR_STD, observedDist)

                new_prob = old_prob * emission_prob
                self.belief.setProb(row, col, new_prob)

        # Normalize the belief distribution
        self.belief.normalize()

    def elapseTime(self) -> None:
        """
        Updates the belief distribution based on the transition model as time passes.
        """
        if self.skipElapse:
            return
            
        new_belief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)

        for (oldTile, newTile), prob in self.transProb.items():
            old_prob = self.belief.getProb(oldTile[0], oldTile[1])
            trans_prob = old_prob * prob

            current_new_prob = new_belief.getProb(newTile[0], newTile[1])
            new_belief.setProb(newTile[0], newTile[1], current_new_prob + trans_prob)

        new_belief.normalize()
        self.belief = new_belief

    def getBelief(self) -> Belief:
        return self.belief


class ParticleFilter(object):
    """
    Maintains and updates a belief distribution using a set of particles.
    """
    NUM_PARTICLES = 200

    def __init__(self, numRows: int, numCols: int):
        self.belief = util.Belief(numRows, numCols)

        # Load transition probabilities into a dictionary
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if oldTile not in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(int)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]

        # Initialize particles randomly
        self.particles = collections.defaultdict(int)
        potentialParticles = list(self.transProbDict.keys())
        for _ in range(self.NUM_PARTICLES):
            particleIndex = int(random.random() * len(potentialParticles))
            self.particles[potentialParticles[particleIndex]] += 1

        self.updateBelief()

    def updateBelief(self) -> None:
        """Updates belief distribution based on particle locations and normalizes it."""
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief

    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        """
        Updates particle distribution based on observation (Re-weight & Re-sample).
        """
        weights = collections.defaultdict(float)

        # 1. Re-weight particles based on observation
        for particle in self.particles:
            row, col = particle
            car_x = util.colToX(col)
            car_y = util.rowToY(row)

            dist = math.sqrt((agentX - car_x) ** 2 + (agentY - car_y) ** 2)
            
            # Weight = Particle count * Emission probability (Gaussian PDF)
            weights[particle] = self.particles[particle] * util.pdf(dist, Const.SONAR_STD, observedDist)

        # 2. Re-sample particles
        new_particles = collections.defaultdict(int)
        for _ in range(self.NUM_PARTICLES):
            new_particle = util.weightedRandomChoice(weights)
            new_particles[new_particle] += 1

        self.particles = new_particles
        self.updateBelief()

    def elapseTime(self) -> None:
        """
        Updates particle locations based on the transition model.
        """
        new_particles = collections.defaultdict(int)

        for particle in self.particles:
            num_particles_here = self.particles[particle]
            trans_probs = self.transProbDict.get(particle, None)
            
            if not trans_probs:
                # If no transition probability exists, particles stay in place
                new_particles[particle] += num_particles_here
                continue

            for _ in range(num_particles_here):
                new_particle = util.weightedRandomChoice(trans_probs)
                new_particles[new_particle] += 1

        self.particles = new_particles

    def getBelief(self) -> Belief:
        return self.belief