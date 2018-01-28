#include <iostream>
#include <fstream>
#include "andres/graph/graph.hxx"
#include "andres/graph/complete-graph.hxx"
#include "andres/graph/multicut-lifted/kernighan-lin.hxx"
#include "andres/graph/multicut-lifted/greedy-additive.hxx"

namespace ag = andres::graph;

/**
 * Fails if the file does not exist!
 */
std::ifstream check_if_file_exists(std::string &url) {
  std::ifstream file(url);
  if (!file) {
    std::cout << "file " << url << " seems to not exist..." << std::endl;
    exit(1);
  }
  return file;
}


/**
 *  we need 4 parameters:
    * edges.txt file where the edges and their weights are stored
    * lifted_edges.txt as above, just for lited edges
    * config.txt where the number of nodes is stored
    * output.txt where the resulting graph is exported to

    edges-structure: (textfile)
          ID  ID  edge_cost
  e.g.     1   2   0.31
           1   3   7.2
           1.  4  -1.22

    config-structure (textfile)
      text file with a single integer number: the number of
      bounding boxes (and thus nodes) in the video (graph)
 */
int main(int argc, char** argv) {
  std::cout << "start cabbage graph solver" << std::endl;


  if (argc != 5) {
    std::cout << "the number of arguments was wrong" << std::endl;
    exit(1);
  }

  std::string url_edges = argv[1];
  std::string url_lifted_edges = argv[2];
  std::string url_config = argv[3];
  std::string url_output = argv[4];


  std::ifstream fconfig = check_if_file_exists(url_config);
  std::ifstream fedges = check_if_file_exists(url_edges);
  std::ifstream flifted_edges = check_if_file_exists(url_lifted_edges);

  // figure out how many bounding boxes we have in total
  int count = 0;
  fconfig >> count;

  std::cout << "Create Graph of size " << count << std::endl;

  // build the graph
  ag::Graph<> original_graph(count);
  ag::CompleteGraph<> lifted_graph(count);
  std::vector<double> weights(lifted_graph.numberOfEdges());

  int i, j;
  double ce;

  while (fedges >> i) {
    fedges >> j;
    fedges >> ce;
    //if (i < count) break;
    //if (j < count) continue;
    original_graph.insertEdge(i, j);
    weights[lifted_graph.findEdge(i, j).second] = ce;
  }

  while (flifted_edges >> i) {
    flifted_edges >> j;
    flifted_edges >> ce;
    //if (i < count) break;
    //if (j < count) continue;
    weights[lifted_graph.findEdge(i, j).second] = ce;
  }

  std::cout << "graph is build... attempt to solve it" << std::endl;


  std::vector<char> edge_labels(lifted_graph.numberOfEdges(), 1);
  ag::multicut_lifted::kernighanLin(original_graph, lifted_graph, weights, edge_labels, edge_labels);

  std::cout << "done" << std::endl;

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
