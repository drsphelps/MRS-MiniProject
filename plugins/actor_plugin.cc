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
 * Modified obstacle avoidance actor plugin to just follow a predefined
 * trajectory and rotate in-place where necessary.
*/

#include <functional>
#include <vector>

#include <ignition/math.hh>
#include "gazebo/physics/physics.hh"
#include "actor_plugin.hh"

using namespace gazebo;
GZ_REGISTER_MODEL_PLUGIN(ActorPlugin)

#define WALKING_ANIMATION "walking"

// Predefined trajectory to follow
std::vector<ignition::math::Vector3d> trajectory {
  ignition::math::Vector3d(-1.476290, -2.420533, 1.213800),
  ignition::math::Vector3d(-0.601466, -2.826860, 1.213800),
  ignition::math::Vector3d(0.288568, -2.495720, 1.213800),
  ignition::math::Vector3d(1.258310, -2.756880, 1.213800),
  ignition::math::Vector3d(2.060090, -2.210830, 1.213800),
  ignition::math::Vector3d(1.889310, -1.083480, 1.213800),
  ignition::math::Vector3d(1.332560, 0.119578, 1.213800),
  ignition::math::Vector3d(0.927496, 1.174740, 1.213800),
  ignition::math::Vector3d(-0.087479, 2.001330, 1.213800),
  ignition::math::Vector3d(-1.209840, 1.547180, 1.213800),
  ignition::math::Vector3d(-2.214230, 1.427367, 1.213800),
  ignition::math::Vector3d(-2.980290, 0.804405, 1.213800),
  ignition::math::Vector3d(-2.399970, -1.871060, 1.213800)
};
// Index into the trajectory vector
unsigned int IDX = 0;

/////////////////////////////////////////////////
ActorPlugin::ActorPlugin()
{
}

/////////////////////////////////////////////////
void ActorPlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
  this->sdf = _sdf;
  this->actor = boost::dynamic_pointer_cast<physics::Actor>(_model);
  this->world = this->actor->GetWorld();

  this->connections.push_back(event::Events::ConnectWorldUpdateBegin(
          std::bind(&ActorPlugin::OnUpdate, this, std::placeholders::_1)));

  this->Reset();

  // Read in the animation factor (applied in the OnUpdate function).
  if (_sdf->HasElement("animation_factor"))
    this->animationFactor = _sdf->Get<double>("animation_factor");
  else
    this->animationFactor = 4.5;
}

/////////////////////////////////////////////////
void ActorPlugin::Reset()
{
  // Start from the first target position in the predefined trajectory
  IDX = 0;

  this->velocity = 0.7;
  this->lastUpdate = 0;

  if (this->sdf && this->sdf->HasElement("target"))
    this->target = this->sdf->Get<ignition::math::Vector3d>("target");
  else
    this->target = trajectory[IDX++];

  auto skelAnims = this->actor->SkeletonAnimations();
  if (skelAnims.find(WALKING_ANIMATION) == skelAnims.end())
  {
    gzerr << "Skeleton animation " << WALKING_ANIMATION << " not found.\n";
  }
  else
  {
    // Create custom trajectory
    this->trajectoryInfo.reset(new physics::TrajectoryInfo());
    this->trajectoryInfo->type = WALKING_ANIMATION;
    this->trajectoryInfo->duration = 1.0;

    this->actor->SetCustomTrajectory(this->trajectoryInfo);
  }
}

/////////////////////////////////////////////////
void ActorPlugin::ChooseNewTarget()
{
  this->target = trajectory[IDX];
  IDX = (IDX + 1) % trajectory.size();
}

/////////////////////////////////////////////////
void ActorPlugin::OnUpdate(const common::UpdateInfo &_info)
{
  // Time since last update
  double dt = (_info.simTime - this->lastUpdate).Double();

  // Absolute pose of actor
  ignition::math::Pose3d pose = this->actor->WorldPose();
  // Vector from current position to new target position
  ignition::math::Vector3d pos = this->target - pose.Pos();
  // Current (roll, pitch, yaw) of actor
  ignition::math::Vector3d rpy = pose.Rot().Euler();

  // Distance between current position and target position
  double distance = pos.Length();

  // Choose a new target position if the actor has reached its current
  // target.
  if (distance < 0.1)
  {
    this->ChooseNewTarget();
    pos = this->target - pose.Pos();
  }

  // Compute the yaw orientation (have to add 90 degrees by default to be parallel with the X-axis in Gazebo)
  ignition::math::Angle yaw = atan2(pos.Y(), pos.X()) + 1.5707 - rpy.Z();
  yaw.Normalize();

  // Rotate in place, instead of jumping.
  if (std::abs(yaw.Radian()) > IGN_DTOR(10))
  {
    pose.Rot() = ignition::math::Quaterniond(1.5707, 0, rpy.Z()+
        yaw.Radian()*0.02);
  }
  else
  {
    pose.Pos() += pos * this->velocity * dt;
    pose.Rot() = ignition::math::Quaterniond(1.5707, 0, rpy.Z()+yaw.Radian());
  }

  // Make sure the actor stays within bounds
  pose.Pos().X(std::max(-4.0, std::min(4.0, pose.Pos().X())));
  pose.Pos().Y(std::max(-4.0, std::min(4.0, pose.Pos().Y())));
  pose.Pos().Z(1.2138);

  // Distance traveled is used to coordinate motion with the walking
  // animation
  double distanceTraveled = (pose.Pos() -
      this->actor->WorldPose().Pos()).Length();

  this->actor->SetWorldPose(pose, false, false);
  this->actor->SetScriptTime(this->actor->ScriptTime() +
    (distanceTraveled * this->animationFactor));
  this->lastUpdate = _info.simTime;
}
