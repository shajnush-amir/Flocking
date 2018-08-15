import sys, argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist, cdist
from numpy.linalg import norm

width, height = 1280, 720


class agents:
    """Class that represents Boids simulation"""

    def __init__(self, members):
        """ initialize the Boid simulation"""
        # init position & velocities

        xcor=np.random.uniform(0,width,members)  #generates N random numbers from 0 to width
        ycor = np.random.uniform(0, height, members)
        self.pVector=np.array(list(zip(xcor,ycor))) #generates an array of Nx2

        #col 1 = x-coordinate, col 2 = y-corrdinate

        # normalized random velocities
        theta = 2 * math.pi * np.random.rand(members)
        self.dVector = np.array(list(zip(np.sin(theta), np.cos(theta))))
        self.members = members
        # min dist of approach
        self.distMin = 30.0
        # max magnitude of velocities calculated by "rules"
        self.vrMax = 0.02
        # max maginitude of final velocity
        self.vfMax = 1.5


    def wrap(self):
        """apply boundary conditions"""
        offWindow = 2.0
        for cor in self.pVector:
            #BC for the x-axis
            if cor[0] > width + offWindow:
                cor[0] = - offWindow
            if cor[0] < - offWindow:
                cor[0] = width + offWindow
            #BC for the y-axis
            if cor[1] > height + offWindow:
                cor[1] = - offWindow
            if cor[1] < - offWindow:
                cor[1] = height + offWindow

    def flocNature(self):
        # apply rule #1 - Separation
        condMet = self.spacMat < 30.0  #it gives a boolean type array with values true or false.

        #D.sum(axis=1) gives a row matrix by summing the column values of D.
        dVector = self.pVector * condMet.sum(axis=1).reshape(self.members, 1) - condMet.dot(self.pVector)
        self.limit(dVector, self.vrMax)
        # different distance threshold
        #D = self.distMatrix < 50.0

        # apply rule #2 - Alignment
        d2Vector = condMet.dot(self.dVector)
        self.limit(d2Vector, self.vrMax)
        dVector += d2Vector;

        # apply rule #3 - Cohesion
        d3Vector = condMet.dot(self.pVector) - self.pVector
        self.limit(d3Vector, self.vrMax)
        dVector += d3Vector

        return dVector


    def Controller(self, frameCount, body, head):
        """Update the simulation by one time step."""
        # get pairwise distances
        # gets the distance from each member and itself
        self.spacMat = squareform(pdist(self.pVector))

        # apply rules:
        self.dVector += self.flocNature()  #updating velocity according to the 3 rules.
        self.limit(self.dVector, self.vfMax) # setting the limit up to max velocity.
        self.pVector += self.dVector # updating position according to the velocity
        self.wrap()

        # update data
        body.set_data(self.pVector.reshape(2 * self.members)[::2],
                     self.pVector.reshape(2 * self.members)[1::2])

        newLocation = self.pVector + 10 * self.dVector / self.vfMax #it determines the position of the head respect to the body

        head.set_data(newLocation.reshape(2 * self.members)[::2],
                      newLocation.reshape(2 * self.members)[1::2])
        #(2*self.N)[::2] selects the x-coordinate values.
        #(2*self.N)[1::2] selects the y-coordinate values.

    def limitVec(self, newLocation, valueMax):
        """limit magnitide of 2D vector"""
        amount = norm(newLocation)
        if amount > valueMax:
            newLocation[0], newLocation[1] = newLocation[0] * valueMax / amount, newLocation[1] * valueMax / amount

    def limit(self, array, valueMax):
        """limit magnitide of 2D vectors in array X to maxValue"""
        for newLocation in array:
            self.limitVec(newLocation, valueMax)




def fbController(frameCount, body, head, agents):
    # print frameNum
    """update function for animation"""
    agents.Controller(frameCount, body, head)
    return body, head


# main() function
def main():
    # use sys.argv if needed
    print('starting boids...')

    parser = argparse.ArgumentParser(description="Implementing Craig Reynold's Boids...")
    # add arguments
    parser.add_argument('--num-boids', dest='members', required=False)
    args = parser.parse_args()

    # number of boids
    members = 50
    if args.members:
        members = int(args.members)

    # create boids
    boids = agents(members)

    # setup plot
    fig = plt.figure()
    ax = plt.axes(xlim=(0, width), ylim=(0, height))

    body, = ax.plot([], [], markersize=10,
                   c='k', marker='8', ls='None')
    head, = ax.plot([], [], markersize=4,
                    c='r', marker='o', ls='None')
    anim = animation.FuncAnimation(fig, fbController, fargs=(body, head, boids),
                                   interval=50)
    plt.show()


# call main
if __name__ == '__main__':
    main()
