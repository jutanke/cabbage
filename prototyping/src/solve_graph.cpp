#include <iostream>
#include "andres/graph/graph.hxx"
#include "andres/graph/complete-graph.hxx"
#include "andres/graph/multicut-lifted/kernighan-lin.hxx"

using namespace andres::graph;

int main() {
  std::cout << "cabbage solve graph" << std::endl;


  Graph<> original_graph(5);
  original_graph.insertEdge(0, 1); // 0
  original_graph.insertEdge(0, 3); // 1
  original_graph.insertEdge(1, 2); // 2
  original_graph.insertEdge(1, 4); // 3
  original_graph.insertEdge(3, 4); // 4

  CompleteGraph<> lifted_graph(5);

  std::vector<double> weights(10);
  weights[lifted_graph.findEdge(0, 1).second] = 10;
  weights[lifted_graph.findEdge(0, 2).second] = -1;
  weights[lifted_graph.findEdge(0, 3).second] = -1;
  weights[lifted_graph.findEdge(0, 4).second] = -1;
  weights[lifted_graph.findEdge(1, 2).second] = 10;
  weights[lifted_graph.findEdge(1, 3).second] = -1;
  weights[lifted_graph.findEdge(1, 4).second] = 4;
  weights[lifted_graph.findEdge(2, 3).second] = -1;
  weights[lifted_graph.findEdge(2, 4).second] = -1;
  weights[lifted_graph.findEdge(3, 4).second] = 10;

 std::vector<char> edge_labels(lifted_graph.numberOfEdges(), 1);
  multicut_lifted::kernighanLin(original_graph, lifted_graph, weights, edge_labels, edge_labels);

  assert(edge_labels[lifted_graph.findEdge(0, 1).second] == 0);
  assert(edge_labels[lifted_graph.findEdge(0, 2).second] == 0);
  assert(edge_labels[lifted_graph.findEdge(0, 3).second] == 1);
  assert(edge_labels[lifted_graph.findEdge(0, 4).second] == 1);
  assert(edge_labels[lifted_graph.findEdge(1, 2).second] == 0);
  assert(edge_labels[lifted_graph.findEdge(1, 3).second] == 1);
  assert(edge_labels[lifted_graph.findEdge(1, 4).second] == 1);
  assert(edge_labels[lifted_graph.findEdge(2, 3).second] == 1);
  assert(edge_labels[lifted_graph.findEdge(2, 4).second] == 1);
  assert(edge_labels[lifted_graph.findEdge(3, 4).second] == 0);

  std::cout << "done!" << std::endl;
}
