#pragma once
#include "BasicGraph.h"
#include <nlohmann/json.hpp>
#include <random>
using json = nlohmann::json;

class KivaGrid :
	public BasicGraph
{
public:
	vector<int> endpoints;
	vector<int> agent_home_locations;
    vector<int> workstations;
    vector<double> workstation_weights;

    // Dummy function to get around inheritance issue
    bool load_map(string fname) { return false; }
    bool load_map(string fname, double left_w_weight, double right_w_weight);
    bool load_map_from_jsonstr(
        std::string json_str, double left_w_weight, double right_w_weight);
    void preprocessing(bool consider_rotation, std::string log_dir); // compute heuristics
    bool get_r_mode() const;
    bool get_w_mode() const;
    double get_avg_task_len(
        unordered_map<int, vector<double>> heuristics) const;

private:
    bool r_mode; // Robot start location is 'r'
    bool w_mode; // Added workstation 'w' to the map and removed 'r'. Robot
                 // will start from endpoints ('e'), and robots' tasks will
                 // then alternate between endpoints and workstations.
                 // Under 'w' mode, 'r' must be removed, and vice versa

    bool load_weighted_map(string fname);
    bool load_unweighted_map(
        string fname, double left_w_weight, double right_w_weight);
    bool load_unweighted_map_from_json(
        json G_json, double left_w_weight, double right_w_weight);
    bool load_weighted_map_from_json(json G_json);
    void infer_sim_mode_from_map(json G_json);
    void check_mode();
};
