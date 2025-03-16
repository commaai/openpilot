/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <optional>

#include "selfdrive/ui/sunnypilot/qt/network/sunnylink/models/user_model.h"
#include "selfdrive/ui/sunnypilot/qt/network/sunnylink/models/role_model.h"
#include "selfdrive/ui/sunnypilot/qt/network/sunnylink/models/sponsor_role_model.h"
#include "selfdrive/ui/ui.h"

class UIStateSP : public UIState {
  Q_OBJECT

public:
  UIStateSP(QObject *parent = 0);
  void updateStatus() override;
  void setSunnylinkRoles(const std::vector<RoleModel> &roles);
  void setSunnylinkDeviceUsers(const std::vector<UserModel> &users);

  inline std::vector<RoleModel> sunnylinkDeviceRoles() const { return sunnylinkRoles; }
  inline bool isSunnylinkAdmin() const {
    return std::any_of(sunnylinkRoles.begin(), sunnylinkRoles.end(), [](const RoleModel &role) {
      return role.roleType == RoleType::Admin;
    });
  }
  inline bool isSunnylinkSponsor() const {
    return std::any_of(sunnylinkRoles.begin(), sunnylinkRoles.end(), [](const RoleModel &role) {
      return role.roleType == RoleType::Sponsor && role.as<SponsorRoleModel>().roleTier != SponsorTier::Free;
    });
  }
  inline SponsorRoleModel sunnylinkSponsorRole() const {
    std::optional<SponsorRoleModel> sponsorRoleWithHighestTier = std::nullopt;
    for (const auto &role : sunnylinkRoles) {
      if(role.roleType != RoleType::Sponsor)
        continue;

      if (auto sponsorRole = role.as<SponsorRoleModel>(); !sponsorRoleWithHighestTier.has_value() || sponsorRoleWithHighestTier->roleTier < sponsorRole.roleTier) {
        sponsorRoleWithHighestTier = sponsorRole;
      }
    }
    return sponsorRoleWithHighestTier.value_or(SponsorRoleModel(RoleType::Sponsor, SponsorTier::Free));
  }
  inline SponsorTier sunnylinkSponsorTier() const {
    return sunnylinkSponsorRole().roleTier;
  }
  inline std::vector<UserModel> sunnylinkDeviceUsers() const { return sunnylinkUsers; }
  inline bool isSunnylinkPaired() const {
    return std::any_of(sunnylinkUsers.begin(), sunnylinkUsers.end(), [](const UserModel &user) {
      return user.user_id.toLower() != "unregisteredsponsor" && user.user_id.toLower() != "temporarysponsor";
    });
  }

signals:
  void sunnylinkRoleChanged(bool subscriber);
  void sunnylinkRolesChanged(std::vector<RoleModel> roles);
  void sunnylinkDeviceUsersChanged(std::vector<UserModel> users);
  void uiUpdate(const UIStateSP &s);

private slots:
  void update() override;

private:
  std::vector<RoleModel> sunnylinkRoles = {};
  std::vector<UserModel> sunnylinkUsers = {};
};

UIStateSP *uiStateSP();
inline UIStateSP *uiState() { return uiStateSP(); };

// device management class
class DeviceSP : public Device {
  Q_OBJECT

public:
  DeviceSP(QObject *parent = 0);
};

DeviceSP *deviceSP();
inline DeviceSP *device() { return deviceSP(); }
