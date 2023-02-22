#!/usr/bin/python3
import math
import heapq

from CS312Graph import *
import time


class NetworkRoutingSolver:
    def __init__(self):

        self.previousNodes = None
        self.shortestPath = None

    def dijkstra(self, graph, start_node_id):
        # Initialize the set of unvisited nodes and their tentative distances
        unvisited = [(0, start_node_id)]
        heapq.heapify(unvisited)
        # Initialize the dictionary of the shortest distances and previous nodes
        shortest_distances = {node.node_id: float('inf') for node in graph.nodes}
        shortest_distances[start_node_id] = 0
        previous_nodes = {node.node_id: None for node in graph.nodes}

        while unvisited:
            # Select the node with the smallest tentative distance
            distance, node_id = heapq.heappop(unvisited)

            # Update the tentative distances of the node's neighbors
            node = graph.nodes[node_id]
            for edge in node.neighbors:
                neighbor = edge.dest
                tentative_distance = shortest_distances[node_id] + edge.length
                if tentative_distance < shortest_distances[neighbor.node_id]:
                    shortest_distances[neighbor.node_id] = tentative_distance
                    previous_nodes[neighbor.node_id] = node_id
                    heapq.heappush(unvisited, (tentative_distance, neighbor.node_id))

        return shortest_distances, previous_nodes

    def initializeNetwork(self, network):
        assert (type(network) == CS312Graph)
        self.network = network

    def getShortestPath(self, destIndex):
        self.dest = destIndex
        shortestPaths = self.shortestPath
        previousNodes = self.previousNodes
        nextNodeIndex = destIndex
        path_edges = []
        total_length = 0
        while previousNodes[nextNodeIndex] is not None:
            firstNodeIndex = nextNodeIndex
            firstNode = self.network.nodes[nextNodeIndex]
            nextNodeIndex = previousNodes[nextNodeIndex]
            secondNode = self.network.nodes[nextNodeIndex]
            edge = CS312GraphEdge(firstNode, secondNode, shortestPaths[firstNodeIndex])
            path_edges.append((edge.src.loc, edge.dest.loc, '{:.0f}'.format(edge.length)))
            total_length += edge.length
        return {'cost': total_length, 'path': path_edges}

    def computeShortestPaths(self, srcIndex, use_heap=False):
        self.source = srcIndex
        t1 = time.time()
        if use_heap:
            self.shortestPath, self.previousNodes = self.dijkstra(self.network, self.source)
        # TODO: RUN DIJKSTRA'S TO DETERMINE SHORTEST PATHS.
        #       ALSO, STORE THE RESULTS FOR THE SUBSEQUENT
        #       CALL TO getShortestPath(dest_index)
        t2 = time.time()
        return (t2 - t1)

    def dijkstra_algo(self, indx):
        distance = {}
        prev = {}
        Q = []
        weight = 0

        for node in self.network.nodes:
            distance[node] = float('inf')
            prev[node] = None
            Q.append(node)
        distance[indx] = 0
        # H = self.make_queue(self.network.nodes, distance)
        while Q:
            u = min(Q, key=lambda v: distance[v])
            Q.remove(u)
            for node in self.network.nodes[u.node_id].neighbors:
                if distance[node.dest] > distance[u] + node.length:
                    distance[node.dest] = distance[u] + node.length
                    prev[node.dest] = u
                    Q.append(node.dest)
        return distance, prev
