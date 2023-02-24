#!/usr/bin/python3

import time
import math
from CS312Graph import *


class NetworkRoutingSolver:
    def __init__(self):
        self.network = None
        self.dest = None
        self.source = None
        self.dicts = Dictionaries()

    def initializeNetwork(self, network):
        assert (type(network) == CS312Graph)
        self.network = network

    # This has a time complexity of 0(E) where E is the number of edges
    # It has a space complexity of O(P) where P is the number of paths
    def getShortestPath(self, destIndex):
        self.dest = destIndex
        path_edges = []
        nodes = self.network.getNodes()
        endNode = nodes[self.dest]
        total_length = self.dicts.get_node_dist(endNode)
        currentNode = endNode
        while self.dicts.get_prev().get(currentNode) is not None:
            nextNode = self.dicts.get_prev().get(currentNode)
            for edge in nextNode.neighbors:
                if edge.dest == currentNode:
                    path_edges.append((nextNode.loc, currentNode.loc, '{:.0f}'.format(edge.length)))
            currentNode = nextNode
        return {'cost': total_length, 'path': path_edges}

    # This has a time complexity of O(N log N) for the heap and O(n^2) for the array
    # It has a space complexity of O(n) for each vertice of teh network
    def computeShortestPaths(self, srcIndex, use_heap=False):
        self.source = srcIndex
        t1 = time.time()
        for x in self.network.nodes:
            self.dicts.set_distance(x, math.inf)
            self.dicts.set_prev(x, None)
        nodes = self.network.getNodes()
        self.dicts.set_distance(nodes[self.source], 0)

        if use_heap:
            H = HeapQueue(self.network, self.dicts)
        else:
            H = ArrayQueue(self.network, self.dicts)
        while len(H.get_queue()) > 0:
            u = H.delete_min()
            for n in u.neighbors:
                v = n.dest
                if self.dicts.get_node_dist(v) > (self.dicts.get_node_dist(u) + n.length):
                    self.dicts.set_distance(v, (self.dicts.get_node_dist(u) + n.length))
                    self.dicts.set_prev(v, u)
                    H.decrease_key(v)
        t2 = time.time()
        return (t2 - t1)


# Asll functions here are constant time since they are simple functions.
# Space complexity is O(n) for each node in the network
class Dictionaries(object):
    def __init__(self):
        self.dist = {}
        self.prev = {}
        self.indexDict = {}

    def get_distance(self):
        return self.dist

    def set_distance(self, node, distance):
        self.dist[node] = distance

    def get_prev(self):
        return self.prev

    def set_prev(self, node, prev):
        self.prev[node] = prev

    def get_node_dist(self, node):
        return self.dist[node]

    def get_node_prev(self, node):
        return self.prev[node]

    def get_index_dict(self):
        return self.indexDict

    def get_node_index(self, node):
        return self.indexDict[node]

    def set_index_dict(self, node, index):
        self.indexDict[node] = index


class ArrayQueue:
    def __init__(self, network, dicts):
        self.network = network
        self.dicts = dicts
        self.queue = []
        self.make_queue()

    def decrease_key(self, index):
        pass

    # Time complexity: 0(n) (for loop)
    def make_queue(self):
        for node in self.network.nodes:
            self.insert(node)

    def insert(self, node):
        self.queue.append(node)

    # This is a O(n) time complexity
    def delete_min(self):
        min_distance = self.dicts.get_node_dist(self.queue[0])
        min_index = 0
        for i in range(len(self.queue)):
            distance = self.dicts.get_node_dist(self.queue[i])
            if distance < min_distance:
                min_distance = distance
                min_index = i
        return self.queue.pop(min_index)

    def get_queue(self):
        return self.queue


class HeapQueue:
    def __init__(self, network, dicts):
        self.network = network
        self.dicts = dicts
        self.H = []
        self.make_queue()

    def decrease_key(self, node):
        return self.bubbleUp(self.dicts.get_node_index(node))

    def make_queue(self):
        nodes = list(self.network.nodes)
        for x in nodes:
            self.insert(x)

    # Insert for time and space complexity is O(n log n)
    def insert(self, node):
        childIndex = len(self.H)
        self.H.append(node)
        self.dicts.set_index_dict(node, childIndex)
        self.bubbleUp(childIndex)

    def delete_min(self):
        u = self.H[0]
        z = self.H[len(self.H) - 1]
        self.H[0] = z
        self.dicts.set_index_dict(self.H[0], 0)
        self.H.pop(len(self.H) - 1)
        self.bubbleDown(0)
        return u

    def get_queue(self):
        return self.H

    # BubbleUp is O(log n) where n is the number of nodes. Space is O(1)
    def bubbleUp(self, childIndex):
        while childIndex > 0:
            parentIndex = (childIndex - 1) // 2
            if self.dicts.get_node_dist(self.H[parentIndex]) > self.dicts.get_node_dist(self.H[childIndex]):
                self.switch(parentIndex, childIndex)
                childIndex = parentIndex
            else:
                break

    # Switch is constant time
    def switch(self, parentIndex, childIndex):
        self.H[parentIndex], self.H[childIndex] = self.H[childIndex], self.H[parentIndex]
        self.dicts.set_index_dict(self.H[parentIndex], parentIndex)
        self.dicts.set_index_dict(self.H[childIndex], childIndex)

    # Time complexity is O(log n) it is propotional to the size of the heap, while the space is O(1).
    def bubbleDown(self, parentIndex):
        while parentIndex < len(self.H):
            leftChildIndex = (2 * parentIndex) + 1
            rightChildIndex = (2 * parentIndex) + 2
            minIndex = parentIndex

            if leftChildIndex < len(self.H) and self.dicts.get_node_dist(
                    self.H[leftChildIndex]) < self.dicts.get_node_dist(self.H[minIndex]):
                minIndex = leftChildIndex

            if rightChildIndex < len(self.H) and self.dicts.get_node_dist(
                    self.H[rightChildIndex]) < self.dicts.get_node_dist(self.H[minIndex]):
                minIndex = rightChildIndex

            if minIndex != parentIndex:
                self.switch(parentIndex, minIndex)
                parentIndex = minIndex
            else:
                break
