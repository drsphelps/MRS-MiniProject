/*
 * Copyright (C) 2016 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/

#ifndef GAZEBO_PLUGINS_ACTORPLUGIN_HH_
#define GAZEBO_PLUGINS_ACTORPLUGIN_HH_

#include <string>
#include <vector>

#include "gazebo/common/Plugin.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/util/system.hh"

namespace gazebo
{
  class GZ_PLUGIN_VISIBLE ActorPlugin : public ModelPlugin
  {
    /// \brief Constructor
    public: ActorPlugin();

    /// \brief Load the actor plugin.
    /// \param[in] _model Pointer to the parent model.
    /// \param[in] _sdf Pointer to the plugin's SDF elements.
    public: virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);

    // Documentation Inherited.
    public: virtual void Reset();

    /// \brief Function that is called every update cycle.
    /// \param[in] _info Timing information
    private: void OnUpdate(const common::UpdateInfo &_info);

    /// \brief Helper function to choose a new target location
    private: void ChooseNewTarget();

    /// \brief Pointer to the parent actor.
    private: physics::ActorPtr actor;

    /// \brief Pointer to the world, for convenience.
    private: physics::WorldPtr world;

    /// \brief Pointer to the sdf element.
    private: sdf::ElementPtr sdf;

    /// \brief Velocity of the actor
    private: ignition::math::Vector3d velocity;

    /// \brief List of connections
    private: std::vector<event::ConnectionPtr> connections;

    /// \brief Current target location
    private: ignition::math::Vector3d target;

    /// \brief Time scaling factor. Used to coordinate translational motion
    /// with the actor's walking animation.
    private: double animationFactor = 1.0;

    /// \brief Time of the last update.
    private: common::Time lastUpdate;

    /// \brief Custom trajectory info.
    private: physics::TrajectoryInfoPtr trajectoryInfo;
  };
}
#endif