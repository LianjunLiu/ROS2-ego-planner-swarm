#ifndef _PLANNER_MANAGER_H_
#define _PLANNER_MANAGER_H_

#include <stdlib.h>

#include <bspline_opt/bspline_optimizer.h>
#include <bspline_opt/uniform_bspline.h>
#include <traj_utils/msg/data_disp.hpp>
#include <plan_env/grid_map.h>
#include <plan_env/obj_predictor.h>
#include <traj_utils/plan_container.hpp>
#include <rclcpp/rclcpp.hpp>
#include <traj_utils/planning_visualization.h>

namespace ego_planner
{

  // Fast Planner Manager
  // Key algorithms of mapping and planning are called

  class EGOPlannerManager
  {
    // SECTION stable
  public:
    EGOPlannerManager();
    ~EGOPlannerManager();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /* main planning interface */
    bool reboundReplan(Eigen::Vector3d start_pt, Eigen::Vector3d start_vel, Eigen::Vector3d start_acc,
                       Eigen::Vector3d end_pt, Eigen::Vector3d end_vel, bool flag_polyInit, bool flag_randomPolyTraj);
    bool EmergencyStop(Eigen::Vector3d stop_pos);
    bool planGlobalTraj(const Eigen::Vector3d &start_pos, const Eigen::Vector3d &start_vel, const Eigen::Vector3d &start_acc,
                        const Eigen::Vector3d &end_pos, const Eigen::Vector3d &end_vel, const Eigen::Vector3d &end_acc);
    bool planGlobalTrajWaypoints(const Eigen::Vector3d &start_pos, const Eigen::Vector3d &start_vel, const Eigen::Vector3d &start_acc,
                                 const std::vector<Eigen::Vector3d> &waypoints, const Eigen::Vector3d &end_vel, const Eigen::Vector3d &end_acc);

    /* Map-aware lookahead.  Runs A* on the inflated map from the vehicle toward
     * goal_pt, clipped to search_radius, and returns a point on the resulting
     * path `lookahead` metres ahead plus the path tangent there.  See the
     * definition for why this is limited to a short receding horizon. */
    bool planLocalTargetAStar(const Eigen::Vector3d &start_pt, const Eigen::Vector3d &goal_pt,
                              double search_radius, double lookahead,
                              Eigen::Vector3d &local_target, Eigen::Vector3d &local_vel);

    /* Full-range map-aware path search: A* over grid_map_'s inflated occupancy
     * from start_pt all the way to goal_pt, in a single horizontal plane.
     *
     * This exists because EGO-Planner's own global reference is a straight line
     * from the vehicle to the goal with no map information in it at all
     * (planGlobalTraj only fits the endpoints).  That is fine when the upstream
     * mission hands over a route that already goes round the obstacles, and
     * hopeless when it does not: over the Penglai pillar field the straight line
     * reaches the goal through a gap the vehicle does not fit through, and a
     * planner whose lookahead is sampled off that line has no mechanism to make
     * the lateral detour that would find a wider one.
     *
     * Deliberately *not* built on path_searching::AStar.  That class allocates a
     * dense pool of individually heap-allocated GridNodes at initGridMap time
     * (100^3 nodes, tens of MB) and confines every search to +-(pool/2 - 1)
     * cells around the query midpoint -- +-7.35 m at a 0.15 m step.  Enlarging
     * the pool to cover a 20 m mission means millions of allocations, and
     * shrinking the step instead loses the ~3-cell channels.  A flat
     * std::vector over the map's own voxel grid costs a few MB, has no centre
     * bias, and can be sized to the map.
     *
     * Planar by design.  The Penglai pillars are 6-12 m tall against a ceiling
     * at virtual_ceil_height, so there is no over-the-top route and the flown
     * profile is level at cruise altitude; a 3D search would only add a z axis
     * the vehicle may not spend, and the caller takes z from the global
     * polynomial regardless.  If the mission ever needs a climb over an
     * obstacle this must become 3D.
     *
     * Returns false when no route exists in the current map.  The caller has to
     * treat that as "no opinion", not as "fly straight": an unmapped map is
     * indistinguishable from a clear one, and the map only ever grows.
     *
     * On success `path` holds the corner points of the route, *including* the
     * snapped start and goal and with the 0.15 m staircase already pulled out.
     * The corners are on the search plane, so their z is the caller's
     * start_pt.z(); the caller supplies the real goal itself, since the goal is
     * snapped to a free cell and can sit up to a snap radius from where the
     * mission actually asked to go. */
    bool planGlobalPathAStar(const Eigen::Vector3d &start_pt, const Eigen::Vector3d &goal_pt,
                             std::vector<Eigen::Vector3d> &path);

    /* Last A* path searched by planLocalTargetAStar(), for diagnostics. */
    std::vector<Eigen::Vector3d> last_astar_path_;

    /* The A* work lists, kept as members so the search does not allocate a few
     * MB of scratch off the heap on every call.  Both are sized to the map's
     * own voxel grid and reset per call. */
    std::vector<float> global_astar_g_;
    std::vector<int> global_astar_from_;

    /* Clearance, in cells, that the last planGlobalPathAStar() route was proven
     * to hold beyond the inflation -- i.e. how far down the ladder it had to go
     * before a route existed.  -1 if the last call returned false.  Centre
     * clearance of that route is at least obstacles_inflation +
     * last_global_clearance_cells_ * resolution; the caller logs it. */
    int last_global_clearance_cells_ = -1;

    void initPlanModules(rclcpp::Node::SharedPtr &node, PlanningVisualization::Ptr vis = NULL);

    void deliverTrajToOptimizer(void) { bspline_optimizer_->setSwarmTrajs(&swarm_trajs_buf_); };

    void setDroneIdtoOpt(void) { bspline_optimizer_->setDroneId(pp_.drone_id); }

    double getSwarmClearance(void) { return bspline_optimizer_->getSwarmClearance(); }

    bool checkCollision(int drone_id);
    

    PlanParameters pp_;
    LocalTrajData local_data_;
    GlobalTrajData global_data_;
    GridMap::Ptr grid_map_;
    fast_planner::ObjPredictor::Ptr obj_predictor_;    
    SwarmTrajData swarm_trajs_buf_;

  private:
    rclcpp::Node::SharedPtr node_;
    /* main planning algorithms & modules */
    PlanningVisualization::Ptr visualization_;

    // ros::Publisher obj_pub_; //zx-todo 

    BsplineOptimizer::Ptr bspline_optimizer_;

    int continous_failures_count_{0};

    void updateTrajInfo(const UniformBspline &position_traj, const rclcpp::Time time_now);

    void reparamBspline(UniformBspline &bspline, vector<Eigen::Vector3d> &start_end_derivative, double ratio, Eigen::MatrixXd &ctrl_pts, double &dt,
                        double &time_inc);

    bool refineTrajAlgo(UniformBspline &traj, vector<Eigen::Vector3d> &start_end_derivative, double ratio, double &ts, Eigen::MatrixXd &optimal_control_points);

    // !SECTION stable

    // SECTION developing

  public:
    typedef unique_ptr<EGOPlannerManager> Ptr;

    // !SECTION
  };
} // namespace ego_planner

#endif
