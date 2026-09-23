// #include <fstream>
#include <ego_planner/planner_manager.h>
#include <chrono>
#include <limits>
#include <queue>
#include <thread>
#include "visualization_msgs/msg/marker.hpp" // zx-todo

namespace ego_planner
{

  EGOPlannerManager::EGOPlannerManager() {}

  EGOPlannerManager::~EGOPlannerManager() {}

  void EGOPlannerManager::initPlanModules(rclcpp::Node::SharedPtr &node, PlanningVisualization::Ptr vis)
  {
    node_ = node;
    node->declare_parameter("manager/max_vel", -1.0);
    node->declare_parameter("manager/max_acc", -1.0);
    node->declare_parameter("manager/traj_validation_window", 2.0);
    node->declare_parameter("manager/max_jerk", -1.0);
    node->declare_parameter("manager/feasibility_tolerance", 0.0);
    node->declare_parameter("manager/control_points_distance", -1.0);
    node->declare_parameter("manager/planning_horizon", 5.0);
    node->declare_parameter("manager/use_distinctive_trajs", false);
    node->declare_parameter("manager/drone_id", -1);

    node->get_parameter("manager/max_vel", pp_.max_vel_);
    node->get_parameter("manager/max_acc", pp_.max_acc_);
    node->get_parameter("manager/traj_validation_window", pp_.traj_validation_window_);
    node->get_parameter("manager/max_jerk", pp_.max_jerk_);
    node->get_parameter("manager/feasibility_tolerance", pp_.feasibility_tolerance_);
    node->get_parameter("manager/control_points_distance", pp_.ctrl_pt_dist);
    node->get_parameter("manager/planning_horizon", pp_.planning_horizen_);
    node->get_parameter("manager/use_distinctive_trajs", pp_.use_distinctive_trajs);
    node->get_parameter("manager/drone_id", pp_.drone_id);

    local_data_.traj_id_ = 0;
    grid_map_.reset(new GridMap);
    // grid_map_->initMap(nh);
    grid_map_->initMap(node);

    bspline_optimizer_.reset(new BsplineOptimizer);
    // bspline_optimizer_->setParam(nh);
    bspline_optimizer_->setParam(node);
    bspline_optimizer_->setEnvironment(grid_map_, obj_predictor_);
    bspline_optimizer_->a_star_.reset(new AStar);
    bspline_optimizer_->a_star_->initGridMap(grid_map_, Eigen::Vector3i(100, 100, 100));

    visualization_ = vis;
  }

  bool EGOPlannerManager::planLocalTargetAStar(const Eigen::Vector3d &start_pt, const Eigen::Vector3d &goal_pt,
                                               double search_radius, double lookahead,
                                               Eigen::Vector3d &local_target, Eigen::Vector3d &local_vel)
  {
    last_astar_path_.clear();

    Eigen::Vector3d to_goal = goal_pt - start_pt;
    const double goal_dist = to_goal.norm();
    if (goal_dist < 1e-3)
      return false;
    to_goal /= goal_dist;

    // AStar's node pool is a fixed cube whose centre is the *midpoint of the
    // query* (AstarSearch does `center_ = (start_pt + end_pt) / 2`) and whose
    // neighbours are limited to index 1..pool_size-2, i.e. +-(pool/2 - 1)
    // cells.  The pool here is the 100^3 one the optimiser already allocates,
    // so at a 0.15 m step the search is confined to +-7.35 m around the
    // midpoint and the far goal (15 m away at spawn) cannot be handed in
    // directly -- Coord2Index would reject it.  Clip the endpoint to
    // search_radius instead and take the lookahead off the path we get.
    //
    // The radius is tried in descending order (6.0, 3.6, 2.0 m here).  When
    // the vehicle is standing at the mouth of the pillar lattice, the point
    // 6 m ahead can sit in a pocket the current map cannot route out of, while
    // a 2 m reach is free; a shorter but *reachable* target still gives the
    // optimiser a direction that goes around rather than into the obstacle.
    //
    // A *failed* search is not cheap: AstarSearch only gives up once it has
    // exhausted the reachable set inside the pool, which takes its own 0.2 s
    // hard limit.  The ladder is therefore capped by wall-clock time as well as
    // by radius -- this call sits on the FSM timer, whose replan period is
    // 0.5 s, so it must not be allowed to overrun.
    const double kStep = 0.15; // == grid_map/resolution: the pinch channel is only ~3 cells wide
    static const double kRadiusLadder[] = {1.0, 0.6, 1.0 / 3.0};
    const auto t_start = std::chrono::steady_clock::now();
    bool found = false;
    for (double frac : kRadiusLadder)
    {
      const double r = search_radius * frac;
      if (r < 2.5 * kStep)
        break;
      Eigen::Vector3d search_end = start_pt + to_goal * std::min(r, goal_dist);
      if (bspline_optimizer_->a_star_->AstarSearch(kStep, start_pt, search_end))
      {
        found = true;
        break;
      }
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() > 0.35)
        break;
    }
    if (!found)
      return false;

    last_astar_path_ = bspline_optimizer_->a_star_->getPath();
    if (last_astar_path_.size() < 2)
      return false;

    // Walk outward along the path and stop at the first point `lookahead` of
    // arc length from the vehicle.  A path shorter than that is used whole:
    // every point on an A* path is a free cell by construction, so the end is a
    // valid B-spline endpoint either way.
    size_t i_hit = last_astar_path_.size() - 1;
    double s = 0.0;
    for (size_t i = 1; i < last_astar_path_.size(); ++i)
    {
      s += (last_astar_path_[i] - last_astar_path_[i - 1]).norm();
      if (s >= lookahead)
      {
        i_hit = i;
        break;
      }
    }

    local_target = last_astar_path_[i_hit];

    // End velocity along the path tangent at that point, at the manager's
    // cruise speed.  Handing the optimiser a velocity that points somewhere
    // other than the target is worse than handing it none: the end-derivative
    // constraint would bend the curve back toward the straight-line direction
    // this whole routine exists to replace.
    //
    // Guard the normalisation: a zero-length difference would make
    // normalized() return NaN, and a NaN end-derivative reaches PX4 as a NaN
    // setpoint.  Fall back to the chord from the vehicle, then to zero.
    Eigen::Vector3d tangent = (i_hit + 1 < last_astar_path_.size())
                                  ? (last_astar_path_[i_hit + 1] - last_astar_path_[i_hit])
                                  : (last_astar_path_[i_hit] - last_astar_path_[i_hit - 1]);
    if (tangent.norm() < 1e-6)
      tangent = local_target - start_pt;
    local_vel = Eigen::Vector3d::Zero();
    if (tangent.norm() >= 1e-6)
      local_vel = tangent.normalized() * pp_.max_vel_;

    return true;
  }

  bool EGOPlannerManager::planGlobalPathAStar(const Eigen::Vector3d &start_pt, const Eigen::Vector3d &goal_pt,
                                              std::vector<Eigen::Vector3d> &path)
  {
    path.clear();
    last_global_clearance_cells_ = -1;

    const double res = grid_map_->getResolution();
    Eigen::Vector3d map_ori, map_size;
    grid_map_->getRegion(map_ori, map_size);
    if (res <= 0.0)
      return false;
    const int nx = static_cast<int>(std::floor(map_size.x() / res));
    const int ny = static_cast<int>(std::floor(map_size.y() / res));
    if (nx < 3 || ny < 3)
      return false;

    // Search in the vehicle's own altitude plane.  The pillars are 6-12 m tall
    // against a virtual ceiling at virtual_ceil_height, so there is no
    // over-the-top route and the mission is flown level: a 3D search would
    // spend its whole budget on a z axis the vehicle never uses.  The 1 m arena
    // walls sit *below* this plane and so are simply absent from the slice --
    // which is exactly what makes leaving the arena over them a legal move
    // rather than an optimistic one.
    const double plane_z = start_pt.z();

    auto index_of = [&](const Eigen::Vector3d &p, int &i, int &j) {
      i = static_cast<int>(std::floor((p.x() - map_ori.x()) / res));
      j = static_cast<int>(std::floor((p.y() - map_ori.y()) / res));
      return i >= 0 && i < nx && j >= 0 && j < ny;
    };
    auto cell_centre = [&](int i, int j) {
      return Eigen::Vector3d(map_ori.x() + (i + 0.5) * res,
                             map_ori.y() + (j + 0.5) * res, plane_z);
    };

    // The plane's inflated occupancy, read once.  getInflateOccupancy() returns
    // -1 both outside the map and inside the inflated band, and every caller
    // reads -1 as an obstacle, so `!= 0` is the whole test.
    std::vector<unsigned char> occ(static_cast<size_t>(nx) * ny, 1);
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        occ[static_cast<size_t>(i) * ny + j] =
            (grid_map_->getInflateOccupancy(cell_centre(i, j)) == 0) ? 0 : 1;

    // Snapping an endpoint to the nearest cell the given grid calls free,
    // instead of rejecting it when it lands in an occupied one.  They routinely
    // do: the vehicle's own odometry can sit inside an inflated cell while it
    // is hugging an obstacle, and the goal is wherever the mission asked for.
    // Refusing to plan would be reporting a fact about the query point rather
    // than about the map.  The ring is searched outward so the snap is a
    // nearest-move, not a scan-order artefact.
    auto snap = [&](const std::vector<unsigned char> &grid, const Eigen::Vector3d &p, int &i, int &j) {
      if (!index_of(p, i, j))
        return false;
      if (grid[static_cast<size_t>(i) * ny + j] == 0)
        return true;
      for (int r = 1; r <= 12; ++r) // 12 cells = 1.8 m at res 0.15
      {
        for (int di = -r; di <= r; ++di)
          for (int dj = -r; dj <= r; ++dj)
          {
            if (std::max(std::abs(di), std::abs(dj)) != r) // the ring only
              continue;
            const int a = i + di, b = j + dj;
            if (a < 0 || a >= nx || b < 0 || b >= ny)
              continue;
            if (grid[static_cast<size_t>(a) * ny + b] == 0)
            {
              i = a;
              j = b;
              return true;
            }
          }
      }
      return false;
    };

    // The clearance ladder, in cells beyond what the inflation already
    // guarantees.  This is the part that matters, and it is here because plain
    // A* is the wrong objective for this map.
    //
    // A* minimises *distance*.  On the inflated map it therefore returns the
    // straight line whenever the straight line is free -- and at
    // obstacles_inflation 0.30 m it is: the corridor the direct line threads is
    // 0.70 m wide, so its centre clears the inflated cells by 0.05 m and the
    // search sees nothing wrong with it.  What the search cannot see is that
    // 0.35 m of centre clearance is *less than the 0.386 m rotor envelope*, so
    // the route it happily returns is one the vehicle does not fit through.
    // "Is there a path" is the wrong question; "how wide is it" is the right
    // one, and that is a bottleneck objective, not a shortest-path one.
    //
    // Rather than implement a bottleneck search, ask A* the right question:
    // search on the map with the obstacles grown by R cells, so any path it
    // returns is guaranteed R cells clear, and try R from large to small.  The
    // first R that yields a path is the widest route the map supports, and the
    // ladder degrades to the old behaviour at R = 0 rather than failing.
    // Growing the obstacles costs nothing extra: it is an erosion of the free
    // set by a square of radius R, which is separable and two linear passes.
    //
    // R = 1 gives 0.30 + 0.15 = 0.45 m of centre clearance, over the 0.386 m
    // envelope; R = 2 gives 0.60 m.  Larger R is not always better: it is the
    // *first* R that works, so a high R is preferred only while the map can
    // still support it -- and when it cannot, the field is sealed at that
    // radius and the route goes round the outside instead, which is the
    // 2.050 m corridor and a better answer than any R that threads the pillars.
    static const int kClearanceLadder[] = {3, 2, 1, 0};

    std::vector<unsigned char> safe;
    int si = 0, sj = 0, gi = 0, gj = 0;
    if (!snap(occ, start_pt, si, sj))
      return false;

    for (int R : kClearanceLadder)
    {
      // Erode the free set by a square of radius R: a cell survives only if
      // every cell within R of it is free.  Two 1-D passes, so the cost does
      // not grow with R.
      safe = occ;
      if (R > 0)
      {
        std::vector<unsigned char> tmp(safe.size(), 0);
        for (int i = 0; i < nx; ++i)
          for (int j = 0; j < ny; ++j)
          {
            unsigned char v = 0;
            for (int d = -R; d <= R && !v; ++d)
            {
              const int b = j + d;
              if (b < 0 || b >= ny)
                v = 1;
              else
                v = safe[static_cast<size_t>(i) * ny + b];
            }
            tmp[static_cast<size_t>(i) * ny + j] = v;
          }
        for (int j = 0; j < ny; ++j)
          for (int i = 0; i < nx; ++i)
          {
            unsigned char v = 0;
            for (int d = -R; d <= R && !v; ++d)
            {
              const int a = i + d;
              if (a < 0 || a >= nx)
                v = 1;
              else
                v = tmp[static_cast<size_t>(a) * ny + j];
            }
            safe[static_cast<size_t>(i) * ny + j] = v;
          }
      }

      // The goal must lie in the eroded set -- that is the point of R -- but
      // the *start* is where the vehicle already is, and demanding R of
      // clearance at the start would make the search fail exactly when the
      // vehicle is hugging a pillar, which is when it is needed most.  So the
      // start is taken from the base grid and only the route has to be safe.
      int a = gi, b = gj;
      if (!snap(safe, goal_pt, a, b))
        continue;
      gi = a;
      gj = b;

      const int start_id = si * ny + sj;
      const int goal_id = gi * ny + gj;
      if (start_id == goal_id)
        continue;

      std::vector<float> &g = global_astar_g_;
      std::vector<int> &from = global_astar_from_;
      g.assign(static_cast<size_t>(nx) * ny, std::numeric_limits<float>::infinity());
      from.assign(static_cast<size_t>(nx) * ny, -1);

      const Eigen::Vector3d goal_cell = cell_centre(gi, gj);
      auto heuristic = [&](int i, int j) {
        return static_cast<float>((cell_centre(i, j) - goal_cell).norm());
      };

      // Min-heap on f = g + h.  Entries are removed lazily: a popped node whose
      // recorded g is better than the one this entry was queued with is stale
      // and skipped, which is cheaper than a decrease-key and is the standard
      // trade.
      typedef std::pair<float, int> OpenNode;
      std::priority_queue<OpenNode, std::vector<OpenNode>, std::greater<OpenNode>> open;
      g[start_id] = 0.0f;
      open.push(OpenNode(heuristic(si, sj), start_id));

      static const int kDi[8] = {1, -1, 0, 0, 1, 1, -1, -1};
      static const int kDj[8] = {0, 0, 1, -1, 1, -1, 1, -1};

      bool found = false;
      while (!open.empty())
      {
        const OpenNode top = open.top();
        open.pop();
        const int id = top.second;
        if (id == goal_id)
        {
          found = true;
          break;
        }
        const int i = id / ny, j = id % ny;
        if (top.first > g[id] + heuristic(i, j) + 1e-4f)
          continue; // stale entry, superseded by a better g
        for (int k = 0; k < 8; ++k)
        {
          const int ni = i + kDi[k], nj = j + kDj[k];
          if (ni < 0 || ni >= nx || nj < 0 || nj >= ny)
            continue;
          const int nid = ni * ny + nj;
          if (safe[nid] != 0)
            continue;
          // No corner cutting.  A diagonal step is allowed only when both of
          // the cells it passes between are also safe; without this the path
          // can slip through the diagonal contact point of two grown
          // obstacles, which is precisely the gap the growth exists to seal.
          if (kDi[k] != 0 && kDj[k] != 0 &&
              (safe[static_cast<size_t>(i + kDi[k]) * ny + j] != 0 ||
               safe[static_cast<size_t>(i) * ny + (j + kDj[k])] != 0))
            continue;
          const float step = (kDi[k] != 0 && kDj[k] != 0)
                                 ? static_cast<float>(res * std::sqrt(2.0))
                                 : static_cast<float>(res);
          const float tentative = g[id] + step;
          if (tentative < g[nid])
          {
            g[nid] = tentative;
            from[nid] = id;
            open.push(OpenNode(tentative + heuristic(ni, nj), nid));
          }
        }
      }
      if (!found || from[goal_id] == -1)
        continue;

      std::vector<int> ids;
      for (int id = goal_id; id != -1; id = from[id])
      {
        ids.push_back(id);
        if (id == start_id)
          break;
      }
      if (ids.size() < 2 || ids.back() != start_id)
        continue;
      std::reverse(ids.begin(), ids.end());

      // Whether a straight chord stays inside the *eroded* free set, sampled at
      // half a cell so a chord cannot thread a gap narrower than the grid can
      // represent.  Pulling the string on the eroded set is what keeps the
      // corners R cells clear: shortcutting on the base map would put them
      // straight back against the inflated boundary.
      auto chord_free = [&](const Eigen::Vector3d &p0, const Eigen::Vector3d &p1) {
        const int n = std::max(1, static_cast<int>(std::ceil((p1 - p0).norm() / (res * 0.5))));
        for (int s = 0; s <= n; ++s)
        {
          int i = 0, j = 0;
          const Eigen::Vector3d p = p0 + (p1 - p0) * (static_cast<double>(s) / n);
          if (!index_of(p, i, j) || safe[static_cast<size_t>(i) * ny + j] != 0)
            return false;
        }
        return true;
      };

      // Pull the string.  The raw A* path is a staircase of 0.15 m steps and
      // planGlobalTrajWaypoints() fits a min-snap polynomial through whatever
      // it is handed, so handing it every stair would be a ~130-column fit for
      // a route whose actual content is a handful of corners.  Walking forward
      // from each anchor to the farthest still-visible cell leaves exactly
      // those corners.
      path.push_back(cell_centre(si, sj));
      size_t anchor = 0;
      while (anchor + 1 < ids.size())
      {
        size_t best = anchor + 1;
        for (size_t probe = anchor + 2; probe < ids.size(); ++probe)
        {
          if (!chord_free(path.back(), cell_centre(ids[probe] / ny, ids[probe] % ny)))
            break;
          best = probe;
        }
        path.push_back(cell_centre(ids[best] / ny, ids[best] % ny));
        anchor = best;
      }

      if (path.size() >= 2)
      {
        last_global_clearance_cells_ = R;
        return true;
      }
      path.clear();
    }

    return false;
  }

  bool EGOPlannerManager::reboundReplan(Eigen::Vector3d start_pt, Eigen::Vector3d start_vel,
                                        Eigen::Vector3d start_acc, Eigen::Vector3d local_target_pt,
                                        Eigen::Vector3d local_target_vel, bool flag_polyInit, bool flag_randomPolyTraj)
  {
    static int count = 0;
    printf("\033[47;30m\n[drone %d replan %d]==============================================\033[0m\n", pp_.drone_id, count++);

    if ((start_pt - local_target_pt).norm() < 0.2)
    {
      cout << "Close to goal" << endl;
      continous_failures_count_++;
      return false;
    }

    bspline_optimizer_->setLocalTargetPt(local_target_pt);

    rclcpp::Time t_start = node_->now();
    rclcpp::Duration t_init(0, 0), t_opt(0, 0), t_refine(0, 0);

    /*** STEP 1: INIT
    根据起始点和目标点的距离计算首个时间步长ts,向量的模大于0.1则用1.5倍否则用5倍
    ***/
    double ts = (start_pt - local_target_pt).norm() > 0.1 ? pp_.ctrl_pt_dist / pp_.max_vel_ * 1.5 : pp_.ctrl_pt_dist / pp_.max_vel_ * 5; // pp_.ctrl_pt_dist / pp_.max_vel_ is too tense, and will surely exceed the acc/vel limits
    vector<Eigen::Vector3d> point_set, start_end_derivatives;
    static bool flag_first_call = true, flag_force_polynomial = false;
    bool flag_regenerate = false;
    do
    {
      point_set.clear();
      start_end_derivatives.clear();
      flag_regenerate = false;

      // 这里如果正常进入if（通常为初次生成），则do部分只进行一次，即只清空一次点集；若进入else则有可能对异常情况重置flag_regenerate并再do一次
      if (flag_first_call || flag_polyInit || flag_force_polynomial /*|| ( start_pt - local_target_pt ).norm() < 1.0*/) // Initial path generated from a min-snap traj by order.
      {
        flag_first_call = false;
        flag_force_polynomial = false;
        // 用于存储生成的轨迹
        PolynomialTraj gl_traj;

        double dist = (start_pt - local_target_pt).norm();
        // 判断 速度的平方/加速度 是否大于dist，并决定如何计算时间
        double time = pow(pp_.max_vel_, 2) / pp_.max_acc_ > dist ? sqrt(dist / pp_.max_acc_) : (dist - pow(pp_.max_vel_, 2) / pp_.max_acc_) / pp_.max_vel_ + 2 * pp_.max_vel_ / pp_.max_acc_;

        if (!flag_randomPolyTraj)
        // false生成一段单一的多项式轨迹，true生成一个包含随机插入点的轨迹
        {
          gl_traj = PolynomialTraj::one_segment_traj_gen(start_pt, start_vel, start_acc, local_target_pt, local_target_vel, Eigen::Vector3d::Zero(), time);
        }
        else
        {
          Eigen::Vector3d horizen_dir = ((start_pt - local_target_pt).cross(Eigen::Vector3d(0, 0, 1))).normalized();
          Eigen::Vector3d vertical_dir = ((start_pt - local_target_pt).cross(horizen_dir)).normalized();
          Eigen::Vector3d random_inserted_pt = (start_pt + local_target_pt) / 2 +
                                               (((double)rand()) / RAND_MAX - 0.5) * (start_pt - local_target_pt).norm() * horizen_dir * 0.8 * (-0.978 / (continous_failures_count_ + 0.989) + 0.989) +
                                               (((double)rand()) / RAND_MAX - 0.5) * (start_pt - local_target_pt).norm() * vertical_dir * 0.4 * (-0.978 / (continous_failures_count_ + 0.989) + 0.989);
          Eigen::MatrixXd pos(3, 3);
          pos.col(0) = start_pt;
          pos.col(1) = random_inserted_pt;
          pos.col(2) = local_target_pt;
          Eigen::VectorXd t(2);
          t(0) = t(1) = time / 2;
          gl_traj = PolynomialTraj::minSnapTraj(pos, start_vel, local_target_vel, start_acc, Eigen::Vector3d::Zero(), t);
        }

        double t;
        bool flag_too_far;
        ts *= 1.5; // ts will be divided by 1.5 in the next
        do
        {
          ts /= 1.5;
          point_set.clear();
          flag_too_far = false;
          Eigen::Vector3d last_pt = gl_traj.evaluate(0);
          for (t = 0; t < time; t += ts)
          {
            Eigen::Vector3d pt = gl_traj.evaluate(t);
            if ((last_pt - pt).norm() > pp_.ctrl_pt_dist * 1.5)
            {
              flag_too_far = true;
              break;
            }
            last_pt = pt;
            point_set.push_back(pt);
          }
        } while (flag_too_far || point_set.size() < 7); // To make sure the initial path has enough points.
        t -= ts;
        start_end_derivatives.push_back(gl_traj.evaluateVel(0));
        start_end_derivatives.push_back(local_target_vel);
        start_end_derivatives.push_back(gl_traj.evaluateAcc(0));
        start_end_derivatives.push_back(gl_traj.evaluateAcc(t));
      }
      else // Initial path generated from previous trajectory.
      {

        double t;
        double t_cur = (node_->now() - local_data_.start_time_).seconds();

        vector<double> pseudo_arc_length;
        vector<Eigen::Vector3d> segment_point;
        pseudo_arc_length.push_back(0.0);
        for (t = t_cur; t < local_data_.duration_ + 1e-3; t += ts)
        {
          segment_point.push_back(local_data_.position_traj_.evaluateDeBoorT(t));
          if (t > t_cur)
          {
            pseudo_arc_length.push_back((segment_point.back() - segment_point[segment_point.size() - 2]).norm() + pseudo_arc_length.back());
          }
        }
        t -= ts;

        double poly_time = (local_data_.position_traj_.evaluateDeBoorT(t) - local_target_pt).norm() / pp_.max_vel_ * 2;
        if (poly_time > ts)
        {
          PolynomialTraj gl_traj = PolynomialTraj::one_segment_traj_gen(local_data_.position_traj_.evaluateDeBoorT(t),
                                                                        local_data_.velocity_traj_.evaluateDeBoorT(t),
                                                                        local_data_.acceleration_traj_.evaluateDeBoorT(t),
                                                                        local_target_pt, local_target_vel, Eigen::Vector3d::Zero(), poly_time);

          for (t = ts; t < poly_time; t += ts)
          {
            if (!pseudo_arc_length.empty())
            {
              segment_point.push_back(gl_traj.evaluate(t));
              pseudo_arc_length.push_back((segment_point.back() - segment_point[segment_point.size() - 2]).norm() + pseudo_arc_length.back());
            }
            else
            {
              RCLCPP_ERROR(rclcpp::get_logger("ego_planner"), "pseudo_arc_length is empty, return!");
              continous_failures_count_++;
              return false;
            }
          }
        }

        double sample_length = 0;
        double cps_dist = pp_.ctrl_pt_dist * 1.5; // cps_dist will be divided by 1.5 in the next
        size_t id = 0;
        do
        {
          cps_dist /= 1.5;
          point_set.clear();
          sample_length = 0;
          id = 0;
          while ((id <= pseudo_arc_length.size() - 2) && sample_length <= pseudo_arc_length.back())
          {
            if (sample_length >= pseudo_arc_length[id] && sample_length < pseudo_arc_length[id + 1])
            {
              point_set.push_back((sample_length - pseudo_arc_length[id]) / (pseudo_arc_length[id + 1] - pseudo_arc_length[id]) * segment_point[id + 1] +
                                  (pseudo_arc_length[id + 1] - sample_length) / (pseudo_arc_length[id + 1] - pseudo_arc_length[id]) * segment_point[id]);
              sample_length += cps_dist;
            }
            else
              id++;
          }
          point_set.push_back(local_target_pt);
        } while (point_set.size() < 7); // If the start point is very close to end point, this will help

        start_end_derivatives.push_back(local_data_.velocity_traj_.evaluateDeBoorT(t_cur));
        start_end_derivatives.push_back(local_target_vel);
        start_end_derivatives.push_back(local_data_.acceleration_traj_.evaluateDeBoorT(t_cur));
        start_end_derivatives.push_back(Eigen::Vector3d::Zero());

        if (point_set.size() > pp_.planning_horizen_ / pp_.ctrl_pt_dist * 3) // The initial path is unnormally too long!
        {
          flag_force_polynomial = true;
          flag_regenerate = true;
        }
      }
    } while (flag_regenerate);

    // 将轨迹变为B样条轨迹
    Eigen::MatrixXd ctrl_pts, ctrl_pts_temp;
    UniformBspline::parameterizeToBspline(ts, point_set, start_end_derivatives, ctrl_pts);

    vector<std::pair<int, int>> segments;
    segments = bspline_optimizer_->initControlPoints(ctrl_pts, true);
    // 计算时间差并更新时间
    auto now = node_->now();
    t_init = now - t_start;
    t_start = now;

    /*** STEP 2: OPTIMIZE ***/
    bool flag_step_1_success = false;
    vector<vector<Eigen::Vector3d>> vis_trajs;

    if (pp_.use_distinctive_trajs)
    {
      // cout << "enter" << endl;
      std::vector<ControlPoints> trajs = bspline_optimizer_->distinctiveTrajs(segments);
      cout << "\033[1;33m"
           << "multi-trajs=" << trajs.size() << "\033[1;0m" << endl;

      double final_cost, min_cost = 999999.0;
      for (int i = trajs.size() - 1; i >= 0; i--)
      {
        if (bspline_optimizer_->BsplineOptimizeTrajRebound(ctrl_pts_temp, final_cost, trajs[i], ts))
        {

          cout << "traj " << trajs.size() - i << " success." << endl;

          flag_step_1_success = true;
          if (final_cost < min_cost)
          {
            min_cost = final_cost;
            ctrl_pts = ctrl_pts_temp;
          }

          // visualization
          point_set.clear();
          for (int j = 0; j < ctrl_pts_temp.cols(); j++)
          {
            point_set.push_back(ctrl_pts_temp.col(j));
          }
          vis_trajs.push_back(point_set);
        }
        else
        {
          cout << "traj " << trajs.size() - i << " failed." << endl;
        }
      }

      t_opt = node_->now() - t_start;

      visualization_->displayMultiInitPathList(vis_trajs, 0.2);
    }
    else
    {
      flag_step_1_success = bspline_optimizer_->BsplineOptimizeTrajRebound(ctrl_pts, ts);
      t_opt = node_->now() - t_start;
      // static int vis_id = 0;
      visualization_->displayInitPathList(point_set, 0.2, 0);
    }

    cout << "plan_success=" << flag_step_1_success << endl;
    if (!flag_step_1_success)
    {
      visualization_->displayOptimalList(ctrl_pts, 0);
      continous_failures_count_++;
      return false;
    }

    t_start = node_->now();

    UniformBspline pos = UniformBspline(ctrl_pts, 3, ts);
    pos.setPhysicalLimits(pp_.max_vel_, pp_.max_acc_, pp_.feasibility_tolerance_);

    // DIAGNOSTIC (2026-09-21): report where the *pre-refine* curve first hits
    // the inflated map, so we can tell whether the time reallocation in
    // refineTrajAlgo is what pushes the trajectory into an obstacle or whether
    // the optimizer's own output was already colliding.
    auto first_hit_frac = [&](UniformBspline &curve) -> double {
      const double dur = curve.getTimeSum();
      const int n = std::max(1, static_cast<int>(std::ceil(dur / 0.02)));
      for (int i = 0; i <= n; ++i)
        if (grid_map_->getInflateOccupancy(curve.evaluateDeBoorT(dur * i / n)) != 0)
          return 100.0 * i / n;
      return -1.0;
    };
    const double pre_hit = first_hit_frac(pos);
    printf("[diag] ts=%.3f ctrl_pts=%d duration=%.2f s  pre-refine first hit: %s\n",
           ts, (int)ctrl_pts.cols(), pos.getTimeSum(),
           pre_hit < 0 ? "none" : (std::to_string((int)pre_hit) + "%").c_str());

    /*** STEP 3: REFINE(RE-ALLOCATE TIME) IF NECESSARY ***/
    // Note: Only adjust time in single drone mode. But we still allow drone_0 to adjust its time profile.
    if (pp_.drone_id <= 0)
    {

      double ratio;
      bool flag_step_2_success = true;
      if (!pos.checkFeasibility(ratio, false))
      {
        cout << "Need to reallocate time." << endl;

        Eigen::MatrixXd optimal_control_points;
        flag_step_2_success = refineTrajAlgo(pos, start_end_derivatives, ratio, ts, optimal_control_points);
        if (flag_step_2_success)
          pos = UniformBspline(optimal_control_points, 3, ts);
      }

      if (!flag_step_2_success)
      {
        printf("\033[34mThis refined trajectory hits obstacles. It doesn't matter if appeares occasionally. But if continously appearing, Increase parameter \"lambda_fitness\".\n\033[0m");
        continous_failures_count_++;
        return false;
      }
    }
    else
    {
      static bool print_once = true;
      if (print_once)
      {
        print_once = false;
        RCLCPP_ERROR(rclcpp::get_logger("ego_planner"), "IN SWARM MODE, REFINE DISABLED!");
      }
    }

    // t_refine = ros::Time::now() - t_start;
    t_refine = node_->now() - t_start;

    // The optimizer's partial-trajectory collision checks are not a final
    // acceptance test, but it is only meaningful to validate the part of the
    // curve the vehicle will actually track.
    //
    // EGO-Planner is a receding-horizon planner: it replans every
    // thresh_replan_time / thresh_no_replan_meter, and the B-spline optimizer
    // correspondingly polices only the first 2/3 of the control points
    // (BsplineOptimizer::check_collision_and_rebound stops at
    // end_idx - (end_idx - order_) / 3).  Enforcing collision-freedom over the
    // *whole* curve therefore rejects trajectories that are safe to fly: in
    // this pillar field it rejected 100% of them, always at ~70% along, past
    // the range the optimizer polices -- so the vehicle never left the ground.
    //
    // Enforce the check hard over the execution window (at max_vel 0.4 m/s a
    // 2.0 s window spans ~0.8 m of a 6 m horizon, four times the 0.5 s replan
    // period) and merely report collisions beyond it.
    const double duration = pos.getTimeSum();
    const int samples = std::max(1, static_cast<int>(std::ceil(duration / 0.02)));
    for (int i = 0; i <= samples; ++i) {
      const double t = duration * i / samples;
      Eigen::Vector3d sample = pos.evaluateDeBoorT(t);
      if (grid_map_->getInflateOccupancy(sample) == 0)
        continue;
      if (t <= pp_.traj_validation_window_) {
        RCLCPP_WARN(node_->get_logger(),
                    "Rejecting trajectory: inflated-map collision at %.2f s (%.0f%% along, pt %.2f %.2f %.2f)",
                    t, 100.0 * i / samples, sample.x(), sample.y(), sample.z());
        continous_failures_count_++;
        return false;
      }
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                           "Trajectory collides at %.2f s (%.0f%% along), beyond the %.1f s validation "
                           "window; accepting, the optimizer does not police the tail of the horizon.",
                           t, 100.0 * i / samples, pp_.traj_validation_window_);
      break; // samples are in increasing t, so this is the first collision
    }
    // save planned results
    updateTrajInfo(pos, node_->now());

    static double sum_time = 0;
    static int count_success = 0;

    sum_time += (t_init + t_opt + t_refine).seconds();

    count_success++;

    // cout << "total time:\033[42m" << (t_init + t_opt + t_refine).toSec() << "\033[0m,optimize:" << (t_init + t_opt).toSec() << ",refine:" << t_refine.toSec() << ",avg_time=" << sum_time / count_success << endl;
    cout << "total time:\033[42m" << (t_init + t_opt + t_refine).seconds() << "\033[0m,optimize:" << (t_init + t_opt).seconds() << ",refine:" << t_refine.seconds() << ",avg_time=" << sum_time / count_success << endl;

    // success. YoY
    continous_failures_count_ = 0;
    return true;
  }

  bool EGOPlannerManager::EmergencyStop(Eigen::Vector3d stop_pos)
  {
    Eigen::MatrixXd control_points(3, 6);
    for (int i = 0; i < 6; i++)
    {
      control_points.col(i) = stop_pos;
    }

    updateTrajInfo(UniformBspline(control_points, 3, 1.0), node_->now());

    return true;
  }

  bool EGOPlannerManager::checkCollision(int drone_id)
  {
    // if (local_data_.start_time_.toSec() < 1e9) // It means my first planning has not started
    if (local_data_.traj_id_ == 0)
      return false;

    // double my_traj_start_time = local_data_.start_time_.toSec();
    // double other_traj_start_time = swarm_trajs_buf_[drone_id].start_time_.toSec();
    double my_traj_start_time = local_data_.start_time_.seconds();
    double other_traj_start_time = swarm_trajs_buf_[drone_id].start_time_.seconds();

    double t_start = max(my_traj_start_time, other_traj_start_time);
    double t_end = min(my_traj_start_time + local_data_.duration_ * 2 / 3, other_traj_start_time + swarm_trajs_buf_[drone_id].duration_);

    for (double t = t_start; t < t_end; t += 0.03)
    {
      if ((local_data_.position_traj_.evaluateDeBoorT(t - my_traj_start_time) - swarm_trajs_buf_[drone_id].position_traj_.evaluateDeBoorT(t - other_traj_start_time)).norm() < bspline_optimizer_->getSwarmClearance())
      {
        return true;
      }
    }

    return false;
  }

  bool EGOPlannerManager::planGlobalTrajWaypoints(const Eigen::Vector3d &start_pos, const Eigen::Vector3d &start_vel, const Eigen::Vector3d &start_acc,
                                                  const std::vector<Eigen::Vector3d> &waypoints, const Eigen::Vector3d &end_vel, const Eigen::Vector3d &end_acc)
  {

    // generate global reference trajectory

    vector<Eigen::Vector3d> points;
    points.push_back(start_pos);

    for (size_t wp_i = 0; wp_i < waypoints.size(); wp_i++)
    {
      points.push_back(waypoints[wp_i]);
    }

    double total_len = 0;
    total_len += (start_pos - waypoints[0]).norm();
    for (size_t i = 0; i < waypoints.size() - 1; i++)
    {
      total_len += (waypoints[i + 1] - waypoints[i]).norm();
    }

    // insert intermediate points if too far
    vector<Eigen::Vector3d> inter_points;
    double dist_thresh = max(total_len / 8, 4.0);

    for (size_t i = 0; i < points.size() - 1; ++i)
    {
      inter_points.push_back(points.at(i));
      double dist = (points.at(i + 1) - points.at(i)).norm();

      if (dist > dist_thresh)
      {
        int id_num = floor(dist / dist_thresh) + 1;

        for (int j = 1; j < id_num; ++j)
        {
          Eigen::Vector3d inter_pt =
              points.at(i) * (1.0 - double(j) / id_num) + points.at(i + 1) * double(j) / id_num;
          inter_points.push_back(inter_pt);
        }
      }
    }

    inter_points.push_back(points.back());

    int pt_num = inter_points.size();
    Eigen::MatrixXd pos(3, pt_num);
    for (int i = 0; i < pt_num; ++i)
      pos.col(i) = inter_points[i];

    Eigen::Vector3d zero(0, 0, 0);
    Eigen::VectorXd time(pt_num - 1);
    for (int i = 0; i < pt_num - 1; ++i)
    {
      time(i) = (pos.col(i + 1) - pos.col(i)).norm() / (pp_.max_vel_);
    }

    time(0) *= 2.0;
    time(time.rows() - 1) *= 2.0;

    PolynomialTraj gl_traj;
    if (pos.cols() >= 3)
      gl_traj = PolynomialTraj::minSnapTraj(pos, start_vel, end_vel, start_acc, end_acc, time);
    else if (pos.cols() == 2)
      gl_traj = PolynomialTraj::one_segment_traj_gen(start_pos, start_vel, start_acc, pos.col(1), end_vel, end_acc, time(0));
    else
      return false;

    auto time_now = node_->now();

    global_data_.setGlobalTraj(gl_traj, time_now);

    return true;
  }

  bool EGOPlannerManager::planGlobalTraj(const Eigen::Vector3d &start_pos, const Eigen::Vector3d &start_vel, const Eigen::Vector3d &start_acc,
                                         const Eigen::Vector3d &end_pos, const Eigen::Vector3d &end_vel, const Eigen::Vector3d &end_acc)
  {

    // generate global reference trajectory

    vector<Eigen::Vector3d> points;
    points.push_back(start_pos);
    points.push_back(end_pos);

    // insert intermediate points if too far
    vector<Eigen::Vector3d> inter_points;
    const double dist_thresh = 4.0;

    for (size_t i = 0; i < points.size() - 1; ++i)
    /*挨个读取点并计算点距判断是否需要插点，随后计算插点并写入矩阵，最后根据插点数量生成全局轨迹
      最终返回值为是否规划成功的布尔值 */
    {
      inter_points.push_back(points.at(i));
      double dist = (points.at(i + 1) - points.at(i)).norm();

      if (dist > dist_thresh)
      {
        int id_num = floor(dist / dist_thresh) + 1;

        for (int j = 1; j < id_num; ++j)
        {
          Eigen::Vector3d inter_pt =
              points.at(i) * (1.0 - double(j) / id_num) + points.at(i + 1) * double(j) / id_num;
          inter_points.push_back(inter_pt);
        }
      }
    }

    inter_points.push_back(points.back());

    // write position matrix
    int pt_num = inter_points.size();
    Eigen::MatrixXd pos(3, pt_num);
    for (int i = 0; i < pt_num; ++i)
      pos.col(i) = inter_points[i];

    Eigen::Vector3d zero(0, 0, 0);
    Eigen::VectorXd time(pt_num - 1);
    for (int i = 0; i < pt_num - 1; ++i)
    {
      time(i) = (pos.col(i + 1) - pos.col(i)).norm() / (pp_.max_vel_);
    }

    time(0) *= 2.0;
    time(time.rows() - 1) *= 2.0;

    PolynomialTraj gl_traj;
    if (pos.cols() >= 3)
      gl_traj = PolynomialTraj::minSnapTraj(pos, start_vel, end_vel, start_acc, end_acc, time);
    else if (pos.cols() == 2)
      gl_traj = PolynomialTraj::one_segment_traj_gen(start_pos, start_vel, start_acc, end_pos, end_vel, end_acc, time(0));
    else
      return false;

    auto time_now = node_->now();

    global_data_.setGlobalTraj(gl_traj, time_now);

    return true;
  }

  bool EGOPlannerManager::refineTrajAlgo(UniformBspline &traj, vector<Eigen::Vector3d> &start_end_derivative, double ratio, double &ts, Eigen::MatrixXd &optimal_control_points)
  {
    double t_inc;

    Eigen::MatrixXd ctrl_pts; // = traj.getControlPoint()

    // std::cout << "ratio: " << ratio << std::endl;
    reparamBspline(traj, start_end_derivative, ratio, ctrl_pts, ts, t_inc);

    traj = UniformBspline(ctrl_pts, 3, ts);

    double t_step = traj.getTimeSum() / (ctrl_pts.cols() - 3);
    bspline_optimizer_->ref_pts_.clear();
    for (double t = 0; t < traj.getTimeSum() + 1e-4; t += t_step)
      bspline_optimizer_->ref_pts_.push_back(traj.evaluateDeBoorT(t));

    bool success = bspline_optimizer_->BsplineOptimizeTrajRefine(ctrl_pts, ts, optimal_control_points);

    return success;
  }

  void EGOPlannerManager::updateTrajInfo(const UniformBspline &position_traj, const rclcpp::Time time_now)
  {
    local_data_.start_time_ = time_now;
    local_data_.position_traj_ = position_traj;
    local_data_.velocity_traj_ = local_data_.position_traj_.getDerivative();
    local_data_.acceleration_traj_ = local_data_.velocity_traj_.getDerivative();
    local_data_.start_pos_ = local_data_.position_traj_.evaluateDeBoorT(0.0);
    local_data_.duration_ = local_data_.position_traj_.getTimeSum();
    local_data_.traj_id_ += 1;
  }

  void EGOPlannerManager::reparamBspline(UniformBspline &bspline, vector<Eigen::Vector3d> &start_end_derivative, double ratio,
                                         Eigen::MatrixXd &ctrl_pts, double &dt, double &time_inc)
  {
    double time_origin = bspline.getTimeSum();
    int seg_num = bspline.getControlPoint().cols() - 3;

    bspline.lengthenTime(ratio);
    double duration = bspline.getTimeSum();
    dt = duration / double(seg_num);
    time_inc = duration - time_origin;

    vector<Eigen::Vector3d> point_set;
    for (double time = 0.0; time <= duration + 1e-4; time += dt)
    {
      point_set.push_back(bspline.evaluateDeBoorT(time));
    }
    UniformBspline::parameterizeToBspline(dt, point_set, start_end_derivative, ctrl_pts);
  }

} // namespace ego_planner
