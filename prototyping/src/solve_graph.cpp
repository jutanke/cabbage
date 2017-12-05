#include <iostream>
#include <fstream>
#include "andres/graph/graph.hxx"
#include "andres/graph/complete-graph.hxx"
#include "andres/graph/multicut-lifted/kernighan-lin.hxx"
#include "andres/graph/multicut-lifted/greedy-additive.hxx"

using namespace andres::graph;

int main() {
  std::cout << "cabbage solve graph" << std::endl;

    std::string url_edges = "../edges.txt";
    std::string url_lifted_edges = "../lifted_edges.txt";
    std::string url_config = "../config.txt";

    std::ifstream fconfig(url_config);
    if (!fconfig) {
      std::cout << "no file " << url_config << std::endl;
      exit(1);
    }

    int count = 0;
    fconfig >> count;

    std::cout << "Create Graph with size " << count << std::endl;

    Graph<> original_graph(count);
    CompleteGraph<> lifted_graph(count);

    std::vector<double> weights(lifted_graph.numberOfEdges());

    // --

    std::ifstream fedges(url_edges);
    if (!fedges) {
      std::cout << "no file " << url_edges << std::endl;
      exit(1);
    }

    int i, j;
    double ce;


    while (fedges >> i) {
      fedges >> j;
      fedges >> ce;

      // std::cout << "--------------" << std::endl;
      // std::cout << i << "," << j << "," << ce << std::endl;
      original_graph.insertEdge(i, j);
      weights[lifted_graph.findEdge(i, j).second] = ce;
    }

    std::ifstream flifted_edges(url_lifted_edges);
    if (!flifted_edges) {
      std::cout << "no file " << url_lifted_edges << std::endl;
      exit(1);
    }

    while (flifted_edges >> i) {
      flifted_edges >> j;
      flifted_edges >> ce;
      weights[lifted_graph.findEdge(i, j).second] = ce;
    }

    std::cout << "graph is build... attempt to solve it" << std::endl;

    // std::vector<char> edge_labels(lifted_graph.numberOfEdges(), 1);
    // multicut_lifted::greedyAdditiveEdgeContraction(original_graph, lifted_graph, weights, edge_labels);

    std::vector<char> edge_labels(lifted_graph.numberOfEdges(), 1);
    multicut_lifted::kernighanLin(original_graph, lifted_graph, weights, edge_labels, edge_labels);

    std::cout << "done" << std::endl;

    auto url_output = "result.txt";
    std::ofstream foutput;
    foutput.open(url_output);

    for (i = 0; i < count; ++i) {
      for (j=(i+1); j < count; ++j) {
        foutput << i << " " << j << " ";
        auto qqq = edge_labels[lifted_graph.findEdge(i, j).second] ;
        if(qqq == 0) {
            foutput << "0" << std::endl;
        } else if (qqq == 1) {
          foutput << "1" << std::endl;
        } else {
          std::cout << "nope" << std::endl;
          exit(1);
        }

      }
    }

    foutput.close();

  std::cout << "done!" << std::endl;
}
