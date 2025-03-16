/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#pragma once

#include <QJsonObject>

enum class SponsorTier {
  Free,
  Novice,
  Supporter,
  Contributor,
  Benefactor,
  Guardian,
};

// haha, a role model xD
class SponsorRoleModel final : RoleModel {
public:
  SponsorTier roleTier;

  explicit SponsorRoleModel(const RoleType &roleType, const SponsorTier &roleTier) : RoleModel(roleType), roleTier(roleTier) {}
  explicit SponsorRoleModel(const QJsonObject &json) : RoleModel(json), roleTier(stringToSponsorTier(json["role_tier"].toString())) {}

  [[nodiscard]] QJsonObject toJson() const {
    QJsonObject json = RoleModel::toJson();
    json["role_tier"] = sponsorTierToString(roleTier);
    return json;
  }

  static SponsorTier stringToSponsorTier(const QString &sponsorTierString) {
    const auto sponsorTierStringLower = sponsorTierString.toLower();
    if (sponsorTierStringLower == "guardian")    return SponsorTier::Guardian;
    if (sponsorTierStringLower == "novice")      return SponsorTier::Novice;
    if (sponsorTierStringLower == "supporter")   return SponsorTier::Supporter;
    if (sponsorTierStringLower == "contributor") return SponsorTier::Contributor;
    if (sponsorTierStringLower == "benefactor")  return SponsorTier::Benefactor;

    // Default to Free
    return SponsorTier::Free;
  }

  static QString sponsorTierToString(const SponsorTier &sponsorTier) {
    switch (sponsorTier) {
      case SponsorTier::Guardian:
        return "Guardian";
      case SponsorTier::Novice:
        return "Novice";
      case SponsorTier::Supporter:
        return "Supporter";
      case SponsorTier::Contributor:
        return "Contributor";
      case SponsorTier::Benefactor:
        return "Benefactor";

      default:  // SponsorTier::Free
        return "Free";
    }
  }
  [[nodiscard]] auto getSponsorTierString() const { return sponsorTierToString(roleTier); }

  static QString sponsorTierToColor(const SponsorTier &sponsorTier) {
    switch (sponsorTier) {
      case SponsorTier::Guardian:
        return "gold";
      case SponsorTier::Benefactor:
        return "mediumseagreen";
      case SponsorTier::Contributor:
        return "steelblue";
      case SponsorTier::Supporter:
        return "mediumpurple";
      case SponsorTier::Novice:
        return "white";
      default:  // SponsorTier::Free
        return "silver";
    }
  }
  [[nodiscard]] auto getSponsorTierColor() const { return sponsorTierToColor(roleTier); }
};
