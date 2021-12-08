#include "selfdrive/ui/navd/route_parser.h"

#include <math.h>
#include <QDebug>
#include <QGeoManeuver>
#include <QGeoPath>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QLocale>
#include <QUrlQuery>

#include "selfdrive/common/params.h"

enum CardinalDirection
{
    CardinalN,
    CardinalE,
    CardinalS,
    CardinalW,
};

inline static CardinalDirection azimuthToCardinalDirection4(double azimuth)
{
    azimuth = fmod(azimuth, 360.0);
    if (azimuth < 45.0 || azimuth > 315.0)
        return CardinalN;
    else if (azimuth < 135.0)
        return CardinalE;
    else if (azimuth < 225.0)
        return CardinalS;
    else
        return CardinalW;
}

static QList<QGeoCoordinate> decodePolyline(const QString &polylineString)
{
    QList<QGeoCoordinate> path;
    if (polylineString.isEmpty())
        return path;

    QByteArray data = polylineString.toLatin1();

    bool parsingLatitude = true;

    int shift = 0;
    int value = 0;

    QGeoCoordinate coord(0, 0);

    for (int i = 0; i < data.length(); ++i)
    {
        unsigned char c = data.at(i) - 63;

        value |= (c & 0x1f) << shift;
        shift += 5;

        // another chunk
        if (c & 0x20)
            continue;

        int diff = (value & 1) ? ~(value >> 1) : (value >> 1);

        if (parsingLatitude)
        {
            coord.setLatitude(coord.latitude() + (double)diff / 1e6);
        }
        else
        {
            coord.setLongitude(coord.longitude() + (double)diff / 1e6);
            path.append(coord);
        }

        parsingLatitude = !parsingLatitude;

        value = 0;
        shift = 0;
    }

    return path;
}

static QString cardinalDirection4(CardinalDirection direction)
{
    switch (direction)
    {
    case CardinalN:
        //: Translations exist at https://github.com/Project-OSRM/osrm-text-instructions.
        //: Always used in "Head %1 [onto <street name>]"
        return MapboxRouteParser::tr("North");
    case CardinalE:
        return MapboxRouteParser::tr("East");
    case CardinalS:
        return MapboxRouteParser::tr("South");
    case CardinalW:
        return MapboxRouteParser::tr("West");
    default:
        return QString();
    }
}

static QString exitOrdinal(int exit)
{
    static QList<QString> ordinals;

    if (!ordinals.size())
    {
        ordinals.append(QLatin1String(""));
        //: always used in " and take the %1 exit [onto <street name>]"
        ordinals.append(MapboxRouteParser::tr("first", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("second", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("third", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("fourth", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("fifth", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("sixth", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("seventh", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("eighth", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("ninth", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("tenth", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("eleventh", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("twelfth", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("thirteenth", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("fourteenth", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("fifteenth", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("sixteenth", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("seventeenth", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("eighteenth", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("nineteenth", "roundabout exit"));
        ordinals.append(MapboxRouteParser::tr("twentieth", "roundabout exit"));
    };

    if (exit < 1 || exit > ordinals.size())
        return QString();
    return ordinals[exit];
}

static QString exitDirection(int exit, const QString &wayName)
{
    /*: Always appended to one of the following strings:
        - "Enter the roundabout"
        - "Enter the rotary"
        - "Enter the rotary <rotaryname>"
    */
    static QString directionExit = MapboxRouteParser::tr(" and take the %1 exit");
    static QString directionExitOnto = MapboxRouteParser::tr(" and take the %1 exit onto %2");

    if (exit < 1 || exit > 20)
        return QString();
    if (wayName.isEmpty())
        return directionExit.arg(exitOrdinal(exit));
    else
        return directionExitOnto.arg(exitOrdinal(exit), wayName);
}

static QString instructionArrive(QGeoManeuver::InstructionDirection direction)
{
    switch (direction)
    {
    case QGeoManeuver::DirectionForward:
        return MapboxRouteParser::tr("You have arrived at your destination, straight ahead");
    case QGeoManeuver::DirectionUTurnLeft:
    case QGeoManeuver::DirectionHardLeft:
    case QGeoManeuver::DirectionLeft:
    case QGeoManeuver::DirectionLightLeft:
    case QGeoManeuver::DirectionBearLeft:
        return MapboxRouteParser::tr("You have arrived at your destination, on the left");
    case QGeoManeuver::DirectionUTurnRight:
    case QGeoManeuver::DirectionHardRight:
    case QGeoManeuver::DirectionRight:
    case QGeoManeuver::DirectionLightRight:
    case QGeoManeuver::DirectionBearRight:
        return MapboxRouteParser::tr("You have arrived at your destination, on the right");
    default:
        return MapboxRouteParser::tr("You have arrived at your destination");
    }
}

static QString instructionContinue(const QString &wayName, QGeoManeuver::InstructionDirection direction)
{
    switch (direction)
    {
    case QGeoManeuver::DirectionForward:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Continue straight");
        else
            return MapboxRouteParser::tr("Continue straight on %1").arg(wayName);
    case QGeoManeuver::DirectionHardLeft:
    case QGeoManeuver::DirectionLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Continue left");
        else
            return MapboxRouteParser::tr("Continue left onto %1").arg(wayName);
    case QGeoManeuver::DirectionLightLeft:
    case QGeoManeuver::DirectionBearLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Continue slightly left");
        else
            return MapboxRouteParser::tr("Continue slightly left on %1").arg(wayName);
    case QGeoManeuver::DirectionHardRight:
    case QGeoManeuver::DirectionRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Continue right");
        else
            return MapboxRouteParser::tr("Continue right onto %1").arg(wayName);
    case QGeoManeuver::DirectionLightRight:
    case QGeoManeuver::DirectionBearRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Continue slightly right");
        else
            return MapboxRouteParser::tr("Continue slightly right on %1").arg(wayName);
    case QGeoManeuver::DirectionUTurnLeft:
    case QGeoManeuver::DirectionUTurnRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Make a U-turn");
        else
            return MapboxRouteParser::tr("Make a U-turn onto %1").arg(wayName);
    default:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Continue");
        else
            return MapboxRouteParser::tr("Continue on %1").arg(wayName);
    }
}

static QString instructionDepart(const QJsonObject &maneuver, const QString &wayName)
{
    double bearing = maneuver.value(QLatin1String("bearing_after")).toDouble(-1.0);
    if (bearing >= 0.0)
    {
        if (wayName.isEmpty())
            //: %1 is "North", "South", "East" or "West"
            return MapboxRouteParser::tr("Head %1").arg(cardinalDirection4(azimuthToCardinalDirection4(bearing)));
        else
            return MapboxRouteParser::tr("Head %1 onto %2").arg(cardinalDirection4(azimuthToCardinalDirection4(bearing)), wayName);
    }
    else
    {
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Depart");
        else
            return MapboxRouteParser::tr("Depart onto %1").arg(wayName);
    }
}

static QString instructionEndOfRoad(const QString &wayName, QGeoManeuver::InstructionDirection direction)
{
    switch (direction)
    {
    case QGeoManeuver::DirectionHardLeft:
    case QGeoManeuver::DirectionLeft:
    case QGeoManeuver::DirectionLightLeft:
    case QGeoManeuver::DirectionBearLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the end of the road, turn left");
        else
            return MapboxRouteParser::tr("At the end of the road, turn left onto %1").arg(wayName);
    case QGeoManeuver::DirectionHardRight:
    case QGeoManeuver::DirectionRight:
    case QGeoManeuver::DirectionLightRight:
    case QGeoManeuver::DirectionBearRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the end of the road, turn right");
        else
            return MapboxRouteParser::tr("At the end of the road, turn right onto %1").arg(wayName);
    case QGeoManeuver::DirectionUTurnLeft:
    case QGeoManeuver::DirectionUTurnRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the end of the road, make a U-turn");
        else
            return MapboxRouteParser::tr("At the end of the road, make a U-turn onto %1").arg(wayName);
    case QGeoManeuver::DirectionForward:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the end of the road, continue straight");
        else
            return MapboxRouteParser::tr("At the end of the road, continue straight onto %1").arg(wayName);
    default:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the end of the road, continue");
        else
            return MapboxRouteParser::tr("At the end of the road, continue onto %1").arg(wayName);
    }
}

static QString instructionFerry(const QString &wayName)
{
    QString instruction = MapboxRouteParser::tr("Take the ferry");
    if (!wayName.isEmpty())
        instruction += QLatin1String(" [") + wayName + QLatin1Char(']');

    return instruction;
}

static QString instructionFork(const QString &wayName, QGeoManeuver::InstructionDirection direction)
{
    switch (direction)
    {
    case QGeoManeuver::DirectionHardLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the fork, take a sharp left");
        else
            return MapboxRouteParser::tr("At the fork, take a sharp left onto %1").arg(wayName);
    case QGeoManeuver::DirectionLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the fork, turn left");
        else
            return MapboxRouteParser::tr("At the fork, turn left onto %1").arg(wayName);
    case QGeoManeuver::DirectionLightLeft:
    case QGeoManeuver::DirectionBearLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the fork, keep left");
        else
            return MapboxRouteParser::tr("At the fork, keep left onto %1").arg(wayName);
    case QGeoManeuver::DirectionHardRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the fork, take a sharp right");
        else
            return MapboxRouteParser::tr("At the fork, take a sharp right onto %1").arg(wayName);
    case QGeoManeuver::DirectionRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the fork, turn right");
        else
            return MapboxRouteParser::tr("At the fork, turn right onto %1").arg(wayName);
    case QGeoManeuver::DirectionLightRight:
    case QGeoManeuver::DirectionBearRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the fork, keep right");
        else
            return MapboxRouteParser::tr("At the fork, keep right onto %1").arg(wayName);
    case QGeoManeuver::DirectionUTurnLeft:
    case QGeoManeuver::DirectionUTurnRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Make a U-turn");
        else
            return MapboxRouteParser::tr("Make a U-turn onto %1").arg(wayName);
    case QGeoManeuver::DirectionForward:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the fork, continue straight ahead");
        else
            return MapboxRouteParser::tr("At the fork, continue straight ahead onto %1").arg(wayName);
    default:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the fork, continue");
        else
            return MapboxRouteParser::tr("At the fork, continue onto %1").arg(wayName);
    }
}

static QString instructionMerge(const QString &wayName, QGeoManeuver::InstructionDirection direction)
{
    switch (direction)
    {
    case QGeoManeuver::DirectionUTurnLeft:
    case QGeoManeuver::DirectionHardLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Merge sharply left");
        else
            return MapboxRouteParser::tr("Merge sharply left onto %1").arg(wayName);
    case QGeoManeuver::DirectionLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Merge left");
        else
            return MapboxRouteParser::tr("Merge left onto %1").arg(wayName);
    case QGeoManeuver::DirectionLightLeft:
    case QGeoManeuver::DirectionBearLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Merge slightly left");
        else
            return MapboxRouteParser::tr("Merge slightly left on %1").arg(wayName);
    case QGeoManeuver::DirectionUTurnRight:
    case QGeoManeuver::DirectionHardRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Merge sharply right");
        else
            return MapboxRouteParser::tr("Merge sharply right onto %1").arg(wayName);
    case QGeoManeuver::DirectionRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Merge right");
        else
            return MapboxRouteParser::tr("Merge right onto %1").arg(wayName);
    case QGeoManeuver::DirectionLightRight:
    case QGeoManeuver::DirectionBearRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Merge slightly right");
        else
            return MapboxRouteParser::tr("Merge slightly right on %1").arg(wayName);
    case QGeoManeuver::DirectionForward:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Merge straight");
        else
            return MapboxRouteParser::tr("Merge straight on %1").arg(wayName);
    default:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Merge");
        else
            return MapboxRouteParser::tr("Merge onto %1").arg(wayName);
    }
}

static QString instructionNewName(const QString &wayName, QGeoManeuver::InstructionDirection direction)
{
    switch (direction)
    {
    case QGeoManeuver::DirectionHardLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Take a sharp left");
        else
            return MapboxRouteParser::tr("Take a sharp left onto %1").arg(wayName);
    case QGeoManeuver::DirectionLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Turn left");
        else
            return MapboxRouteParser::tr("Turn left onto %1").arg(wayName);
    case QGeoManeuver::DirectionLightLeft:
    case QGeoManeuver::DirectionBearLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Continue slightly left");
        else
            return MapboxRouteParser::tr("Continue slightly left onto %1").arg(wayName);
    case QGeoManeuver::DirectionHardRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Take a sharp right");
        else
            return MapboxRouteParser::tr("Take a sharp right onto %1").arg(wayName);
    case QGeoManeuver::DirectionRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Turn right");
        else
            return MapboxRouteParser::tr("Turn right onto %1").arg(wayName);
    case QGeoManeuver::DirectionLightRight:
    case QGeoManeuver::DirectionBearRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Continue slightly right");
        else
            return MapboxRouteParser::tr("Continue slightly right onto %1").arg(wayName);
    case QGeoManeuver::DirectionUTurnLeft:
    case QGeoManeuver::DirectionUTurnRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Make a U-turn");
        else
            return MapboxRouteParser::tr("Make a U-turn onto %1").arg(wayName);
    case QGeoManeuver::DirectionForward:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Continue straight");
        else
            return MapboxRouteParser::tr("Continue straight onto %1").arg(wayName);
    default:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Continue");
        else
            return MapboxRouteParser::tr("Continue onto %1").arg(wayName);
    }
}

static QString instructionNotification(const QString &wayName, QGeoManeuver::InstructionDirection direction)
{
    switch (direction)
    {
    case QGeoManeuver::DirectionUTurnLeft:
    case QGeoManeuver::DirectionHardLeft:
    case QGeoManeuver::DirectionLeft:
    case QGeoManeuver::DirectionLightLeft:
    case QGeoManeuver::DirectionBearLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Continue on the left");
        else
            return MapboxRouteParser::tr("Continue on the left on %1").arg(wayName);
    case QGeoManeuver::DirectionUTurnRight:
    case QGeoManeuver::DirectionHardRight:
    case QGeoManeuver::DirectionRight:
    case QGeoManeuver::DirectionLightRight:
    case QGeoManeuver::DirectionBearRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Continue on the right");
        else
            return MapboxRouteParser::tr("Continue on the right on %1").arg(wayName);
    case QGeoManeuver::DirectionForward:
    default:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Continue");
        else
            return MapboxRouteParser::tr("Continue on %1").arg(wayName);
    }
}

static QString instructionOffRamp(const QString &wayName, QGeoManeuver::InstructionDirection direction)
{
    switch (direction)
    {
    case QGeoManeuver::DirectionUTurnLeft:
    case QGeoManeuver::DirectionHardLeft:
    case QGeoManeuver::DirectionLeft:
    case QGeoManeuver::DirectionLightLeft:
    case QGeoManeuver::DirectionBearLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Take the ramp on the left");
        else
            return MapboxRouteParser::tr("Take the ramp on the left onto %1").arg(wayName);
    case QGeoManeuver::DirectionUTurnRight:
    case QGeoManeuver::DirectionHardRight:
    case QGeoManeuver::DirectionRight:
    case QGeoManeuver::DirectionLightRight:
    case QGeoManeuver::DirectionBearRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Take the ramp on the right");
        else
            return MapboxRouteParser::tr("Take the ramp on the right onto %1").arg(wayName);
    case QGeoManeuver::DirectionForward:
    default:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Take the ramp");
        else
            return MapboxRouteParser::tr("Take the ramp onto %1").arg(wayName);
    }
}

static QString instructionOnRamp(const QString &wayName, QGeoManeuver::InstructionDirection direction)
{
    return instructionOffRamp(wayName, direction);
}

static QString instructionPushingBike(const QString &wayName)
{
    if (wayName.isEmpty())
        return MapboxRouteParser::tr("Get off the bike and push");
    else
        return MapboxRouteParser::tr("Get off the bike and push onto %1").arg(wayName);
}

static QString instructionRotary(const QJsonObject &step, const QJsonObject &maneuver, const QString &wayName)
{
    QString instruction;
    QString rotaryName = step.value(QLatin1String("rotary_name")).toString();
    //QString modifier = maneuver.value(QLatin1String("modifier")).toString(); // Apparently not used for rotaries
    int exit = maneuver.value(QLatin1String("exit")).toInt(0);

    //: This string will be prepended to " and take the <nth> exit [onto <streetname>]
    instruction += MapboxRouteParser::tr("Enter the rotary");
    if (!rotaryName.isEmpty())
        instruction += QLatin1Char(' ') + rotaryName;
    instruction += exitDirection(exit, wayName);
    return instruction;
}

static QString instructionRoundabout(const QJsonObject &maneuver, const QString &wayName)
{
    QString instruction;
    //QString modifier = maneuver.value(QLatin1String("modifier")).toString(); // Apparently not used for rotaries
    int exit = maneuver.value(QLatin1String("exit")).toInt(0);

    //: This string will be prepended to " and take the <nth> exit [onto <streetname>]
    instruction += MapboxRouteParser::tr("Enter the roundabout");
    instruction += exitDirection(exit, wayName);
    return instruction;
}

static QString instructionRoundaboutTurn(const QString &wayName, QGeoManeuver::InstructionDirection direction)
{
    switch (direction)
    {
    case QGeoManeuver::DirectionForward:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the roundabout, continue straight");
        else
            return MapboxRouteParser::tr("At the roundabout, continue straight on %1").arg(wayName);
    case QGeoManeuver::DirectionHardLeft:
    case QGeoManeuver::DirectionLeft:
    case QGeoManeuver::DirectionLightLeft:
    case QGeoManeuver::DirectionBearLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the roundabout, turn left");
        else
            return MapboxRouteParser::tr("At the roundabout, turn left onto %1").arg(wayName);
    case QGeoManeuver::DirectionHardRight:
    case QGeoManeuver::DirectionRight:
    case QGeoManeuver::DirectionLightRight:
    case QGeoManeuver::DirectionBearRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the roundabout, turn right");
        else
            return MapboxRouteParser::tr("At the roundabout, turn right onto %1").arg(wayName);
    case QGeoManeuver::DirectionUTurnLeft:
    case QGeoManeuver::DirectionUTurnRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the roundabout, turn around");
        else
            return MapboxRouteParser::tr("At the roundabout, turn around onto %1").arg(wayName);
    default:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("At the roundabout, continue");
        else
            return MapboxRouteParser::tr("At the roundabout, continue onto %1").arg(wayName);
    }
}

static QString instructionTrain(const QString &wayName)
{
    return wayName.isEmpty()
               ? MapboxRouteParser::tr("Take the train")
               : MapboxRouteParser::tr("Take the train [%1]").arg(wayName);
}

static QString instructionTurn(const QString &wayName, QGeoManeuver::InstructionDirection direction)
{
    switch (direction)
    {
    case QGeoManeuver::DirectionForward:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Go straight");
        else
            return MapboxRouteParser::tr("Go straight onto %1").arg(wayName);
    case QGeoManeuver::DirectionHardLeft:
    case QGeoManeuver::DirectionLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Turn left");
        else
            return MapboxRouteParser::tr("Turn left onto %1").arg(wayName);
    case QGeoManeuver::DirectionLightLeft:
    case QGeoManeuver::DirectionBearLeft:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Turn slightly left");
        else
            return MapboxRouteParser::tr("Turn slightly left onto %1").arg(wayName);
    case QGeoManeuver::DirectionHardRight:
    case QGeoManeuver::DirectionRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Turn right");
        else
            return MapboxRouteParser::tr("Turn right onto %1").arg(wayName);
    case QGeoManeuver::DirectionLightRight:
    case QGeoManeuver::DirectionBearRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Turn slightly right");
        else
            return MapboxRouteParser::tr("Turn slightly right onto %1").arg(wayName);
    case QGeoManeuver::DirectionUTurnLeft:
    case QGeoManeuver::DirectionUTurnRight:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Make a U-turn");
        else
            return MapboxRouteParser::tr("Make a U-turn onto %1").arg(wayName);
    default:
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Turn");
        else
            return MapboxRouteParser::tr("Turn onto %1").arg(wayName);
    }
}

static QString instructionUseLane(const QJsonObject &maneuver, const QString &wayName, QGeoManeuver::InstructionDirection direction)
{
    QString laneTypes = maneuver.value(QLatin1String("laneTypes")).toString();
    QString laneInstruction;
    if (laneTypes == QLatin1String("xo") || laneTypes == QLatin1String("xoo") || laneTypes == QLatin1String("xxo"))
        //: "and <instruction direction> [onto <street name>] will be appended to this string. E.g., "Keep right and make a sharp left"
        laneInstruction = QLatin1String("Keep right");
    else if (laneTypes == QLatin1String("ox") || laneTypes == QLatin1String("oox") || laneTypes == QLatin1String("oxx"))
        laneInstruction = QLatin1String("Keep left");
    else if (laneTypes == QLatin1String("xox"))
        laneInstruction = QLatin1String("Use the middle lane");
    else if (laneTypes == QLatin1String("oxo"))
        laneInstruction = QLatin1String("Use the left or the right lane");

    if (laneInstruction.isEmpty())
    {
        if (wayName.isEmpty())
            return MapboxRouteParser::tr("Continue straight");
        else
            return MapboxRouteParser::tr("Continue straight onto %1").arg(wayName);
    }

    switch (direction)
    {
    case QGeoManeuver::DirectionForward:
        if (wayName.isEmpty())
            //: This string will be prepended with lane instructions. E.g., "Use the left or the right lane and continue straight"
            return laneInstruction + MapboxRouteParser::tr(" and continue straight");
        else
            return laneInstruction + MapboxRouteParser::tr(" and continue straight onto %1").arg(wayName);
    case QGeoManeuver::DirectionHardLeft:
        if (wayName.isEmpty())
            return laneInstruction + MapboxRouteParser::tr(" and make a sharp left");
        else
            return laneInstruction + MapboxRouteParser::tr(" and make a sharp left onto %1").arg(wayName);
    case QGeoManeuver::DirectionLeft:
        if (wayName.isEmpty())
            return laneInstruction + MapboxRouteParser::tr(" and turn left");
        else
            return laneInstruction + MapboxRouteParser::tr(" and turn left onto %1").arg(wayName);
    case QGeoManeuver::DirectionLightLeft:
    case QGeoManeuver::DirectionBearLeft:
        if (wayName.isEmpty())
            return laneInstruction + MapboxRouteParser::tr(" and make a slight left");
        else
            return laneInstruction + MapboxRouteParser::tr(" and make a slight left onto %1").arg(wayName);
    case QGeoManeuver::DirectionHardRight:
        if (wayName.isEmpty())
            return laneInstruction + MapboxRouteParser::tr(" and make a sharp right");
        else
            return laneInstruction + MapboxRouteParser::tr(" and make a sharp right onto %1").arg(wayName);
    case QGeoManeuver::DirectionRight:
        if (wayName.isEmpty())
            return laneInstruction + MapboxRouteParser::tr(" and turn right");
        else
            return laneInstruction + MapboxRouteParser::tr(" and turn right onto %1").arg(wayName);
    case QGeoManeuver::DirectionLightRight:
    case QGeoManeuver::DirectionBearRight:
        if (wayName.isEmpty())
            return laneInstruction + MapboxRouteParser::tr(" and make a slight right");
        else
            return laneInstruction + MapboxRouteParser::tr(" and make a slight right onto %1").arg(wayName);
    case QGeoManeuver::DirectionUTurnLeft:
    case QGeoManeuver::DirectionUTurnRight:
        if (wayName.isEmpty())
            return laneInstruction + MapboxRouteParser::tr(" and make a U-turn");
        else
            return laneInstruction + MapboxRouteParser::tr(" and make a U-turn onto %1").arg(wayName);
    default:
        return laneInstruction;
    }
}

static QString instructionText(const QJsonObject &step, const QJsonObject &maneuver, QGeoManeuver::InstructionDirection direction)
{
    QString modifier;
    if (maneuver.value(QLatin1String("modifier")).isString())
        modifier = maneuver.value(QLatin1String("modifier")).toString();
    QString maneuverType;
    if (maneuver.value(QLatin1String("type")).isString())
        maneuverType = maneuver.value(QLatin1String("type")).toString();
    QString wayName = QLatin1String("unknown street");
    if (step.value(QLatin1String("name")).isString())
        wayName = step.value(QLatin1String("name")).toString();

    if (maneuverType == QLatin1String("arrive"))
        return instructionArrive(direction);
    else if (maneuverType == QLatin1String("continue"))
        return instructionContinue(wayName, direction);
    else if (maneuverType == QLatin1String("depart"))
        return instructionDepart(maneuver, wayName);
    else if (maneuverType == QLatin1String("end of road"))
        return instructionEndOfRoad(wayName, direction);
    else if (maneuverType == QLatin1String("ferry"))
        return instructionFerry(wayName);
    else if (maneuverType == QLatin1String("fork"))
        return instructionFork(wayName, direction);
    else if (maneuverType == QLatin1String("merge"))
        return instructionMerge(wayName, direction);
    else if (maneuverType == QLatin1String("new name"))
        return instructionNewName(wayName, direction);
    else if (maneuverType == QLatin1String("notification"))
        return instructionNotification(wayName, direction);
    else if (maneuverType == QLatin1String("off ramp"))
        return instructionOffRamp(wayName, direction);
    else if (maneuverType == QLatin1String("on ramp"))
        return instructionOnRamp(wayName, direction);
    else if (maneuverType == QLatin1String("pushing bike"))
        return instructionPushingBike(wayName);
    else if (maneuverType == QLatin1String("rotary"))
        return instructionRotary(step, maneuver, wayName);
    else if (maneuverType == QLatin1String("roundabout"))
        return instructionRoundabout(maneuver, wayName);
    else if (maneuverType == QLatin1String("roundabout turn"))
        return instructionRoundaboutTurn(wayName, direction);
    else if (maneuverType == QLatin1String("train"))
        return instructionTrain(wayName);
    else if (maneuverType == QLatin1String("turn"))
        return instructionTurn(wayName, direction);
    else if (maneuverType == QLatin1String("use lane"))
        return instructionUseLane(maneuver, wayName, direction);
    else
        return maneuverType + QLatin1String(" to/onto ") + wayName;
}

static QGeoManeuver::InstructionDirection instructionDirection(const QJsonObject &maneuver, MapboxRouteParser::TrafficSide trafficSide)
{
    QString modifier;
    if (maneuver.value(QLatin1String("modifier")).isString())
        modifier = maneuver.value(QLatin1String("modifier")).toString();

    if (modifier.isEmpty())
        return QGeoManeuver::NoDirection;
    else if (modifier == QLatin1String("straight"))
        return QGeoManeuver::DirectionForward;
    else if (modifier == QLatin1String("right"))
        return QGeoManeuver::DirectionRight;
    else if (modifier == QLatin1String("sharp right"))
        return QGeoManeuver::DirectionHardRight;
    else if (modifier == QLatin1String("slight right"))
        return QGeoManeuver::DirectionLightRight;
    else if (modifier == QLatin1String("uturn"))
    {
        switch (trafficSide)
        {
        case MapboxRouteParser::RightHandTraffic:
            return QGeoManeuver::DirectionUTurnLeft;
        case MapboxRouteParser::LeftHandTraffic:
            return QGeoManeuver::DirectionUTurnRight;
        }
        return QGeoManeuver::DirectionUTurnLeft;
    }
    else if (modifier == QLatin1String("left"))
        return QGeoManeuver::DirectionLeft;
    else if (modifier == QLatin1String("sharp left"))
        return QGeoManeuver::DirectionHardLeft;
    else if (modifier == QLatin1String("slight left"))
        return QGeoManeuver::DirectionLightLeft;
    else
        return QGeoManeuver::NoDirection;
}

static QVariantMap parseMapboxVoiceInstruction(const QJsonObject &voiceInstruction)
{
    QVariantMap map;

    if (voiceInstruction.value(QLatin1String("distanceAlongGeometry")).isDouble())
        map.insert(QLatin1String("distance_along_geometry"), voiceInstruction.value(QLatin1String("distanceAlongGeometry")).toDouble());

    if (voiceInstruction.value(QLatin1String("announcement")).isString())
        map.insert(QLatin1String("announcement"), voiceInstruction.value(QLatin1String("announcement")).toString());

    if (voiceInstruction.value(QLatin1String("ssmlAnnouncement")).isString())
        map.insert(QLatin1String("ssml_announcement"), voiceInstruction.value(QLatin1String("ssmlAnnouncement")).toString());

    return map;
}

static QVariantList parseMapboxVoiceInstructions(const QJsonArray &voiceInstructions)
{
    QVariantList list;
    for (const QJsonValue &voiceInstructionValue : voiceInstructions)
    {
        if (voiceInstructionValue.isObject())
            list << parseMapboxVoiceInstruction(voiceInstructionValue.toObject());
    }
    return list;
}

static QVariantMap parseMapboxBannerComponent(const QJsonObject &bannerComponent)
{
    QVariantMap map;

    if (bannerComponent.value(QLatin1String("type")).isString())
        map.insert(QLatin1String("type"), bannerComponent.value(QLatin1String("type")).toString());

    if (bannerComponent.value(QLatin1String("text")).isString())
        map.insert(QLatin1String("text"), bannerComponent.value(QLatin1String("text")).toString());

    if (bannerComponent.value(QLatin1String("abbr")).isString())
        map.insert(QLatin1String("abbr"), bannerComponent.value(QLatin1String("abbr")).toString());

    if (bannerComponent.value(QLatin1String("abbr_priority")).isDouble())
        map.insert(QLatin1String("abbr_priority"), bannerComponent.value(QLatin1String("abbr_priority")).toInt());

    return map;
}

static QVariantList parseMapboxBannerComponents(const QJsonArray &bannerComponents)
{
    QVariantList list;
    for (const QJsonValue &bannerComponentValue : bannerComponents)
    {
        if (bannerComponentValue.isObject())
            list << parseMapboxBannerComponent(bannerComponentValue.toObject());
    }
    return list;
}

static QVariantMap parseMapboxBanner(const QJsonObject &banner)
{
    QVariantMap map;

    if (banner.value(QLatin1String("text")).isString())
        map.insert(QLatin1String("text"), banner.value(QLatin1String("text")).toString());

    if (banner.value(QLatin1String("components")).isArray())
        map.insert(QLatin1String("components"), parseMapboxBannerComponents(banner.value(QLatin1String("components")).toArray()));

    if (banner.value(QLatin1String("type")).isString())
        map.insert(QLatin1String("type"), banner.value(QLatin1String("type")).toString());

    if (banner.value(QLatin1String("modifier")).isString())
        map.insert(QLatin1String("modifier"), banner.value(QLatin1String("modifier")).toString());

    if (banner.value(QLatin1String("degrees")).isDouble())
        map.insert(QLatin1String("degrees"), banner.value(QLatin1String("degrees")).toDouble());

    if (banner.value(QLatin1String("driving_side")).isString())
        map.insert(QLatin1String("driving_side"), banner.value(QLatin1String("driving_side")).toString());

    return map;
}

static QVariantMap parseMapboxBannerInstruction(const QJsonObject &bannerInstruction)
{
    QVariantMap map;

    if (bannerInstruction.value(QLatin1String("distanceAlongGeometry")).isDouble())
        map.insert(QLatin1String("distance_along_geometry"), bannerInstruction.value(QLatin1String("distanceAlongGeometry")).toDouble());

    if (bannerInstruction.value(QLatin1String("primary")).isObject())
        map.insert(QLatin1String("primary"), parseMapboxBanner(bannerInstruction.value(QLatin1String("primary")).toObject()));

    if (bannerInstruction.value(QLatin1String("secondary")).isObject())
        map.insert(QLatin1String("secondary"), parseMapboxBanner(bannerInstruction.value(QLatin1String("secondary")).toObject()));

    if (bannerInstruction.value(QLatin1String("then")).isObject())
        map.insert(QLatin1String("then"), parseMapboxBanner(bannerInstruction.value(QLatin1String("then")).toObject()));

    return map;
}

static QVariantList parseMapboxBannerInstructions(const QJsonArray &bannerInstructions)
{
    QVariantList list;
    for (const QJsonValue &bannerInstructionValue : bannerInstructions)
    {
        if (bannerInstructionValue.isObject())
            list << parseMapboxBannerInstruction(bannerInstructionValue.toObject());
    }
    return list;
}

MapboxRouteParser::MapboxRouteParser(const QString &accessToken, bool useMapboxTextInstructions)
    : m_accessToken(accessToken),
      m_useMapboxTextInstructions(useMapboxTextInstructions),
      trafficSide(Params().getBool("IsRHD") ? MapboxRouteParser::RightHandTraffic : MapboxRouteParser::LeftHandTraffic)
{
}

void MapboxRouteParser::updateSegment(QGeoRouteSegment &segment, const QJsonObject &step, const QJsonObject &maneuver) const
{
    QGeoManeuver m = segment.maneuver();
    QVariantMap extendedAttributes = m.extendedAttributes();
    if (m_useMapboxTextInstructions && maneuver.value(QLatin1String("instruction")).isString())
    {
        QString maneuverInstructionText = maneuver.value(QLatin1String("instruction")).toString();
        if (!maneuverInstructionText.isEmpty())
            m.setInstructionText(maneuverInstructionText);
    }

    if (step.value(QLatin1String("voiceInstructions")).isArray())
        extendedAttributes.insert(QLatin1String("mapbox.voice_instructions"),
                                  parseMapboxVoiceInstructions(step.value(QLatin1String("voiceInstructions")).toArray()));
    if (step.value(QLatin1String("bannerInstructions")).isArray())
        extendedAttributes.insert(QLatin1String("mapbox.banner_instructions"),
                                  parseMapboxBannerInstructions(step.value(QLatin1String("bannerInstructions")).toArray()));

    m.setExtendedAttributes(extendedAttributes);
    segment.setManeuver(m);
}

MapboxRouteSegment MapboxRouteParser::parseStep(const QJsonObject &step, int legIndex, int stepIndex) const
{
    // OSRM Instructions documentation: https://github.com/Project-OSRM/osrm-text-instructions
    // This goes on top of OSRM: https://github.com/Project-OSRM/osrm-backend/blob/master/docs/http.md
    // Mapbox however, includes this in the reply, under "instruction".
    MapboxRouteSegment segment;
    if (!step.value(QLatin1String("maneuver")).isObject())
        return segment;
    QJsonObject maneuver = step.value(QLatin1String("maneuver")).toObject();
    if (!step.value(QLatin1String("duration")).isDouble())
        return segment;
    if (!step.value(QLatin1String("distance")).isDouble())
        return segment;
    if (!step.value(QLatin1String("intersections")).isArray())
        return segment;
    if (!maneuver.value(QLatin1String("location")).isArray())
        return segment;

    double time = step.value(QLatin1String("duration")).toDouble();
    double distance = step.value(QLatin1String("distance")).toDouble();

    QJsonArray position = maneuver.value(QLatin1String("location")).toArray();
    if (position.isEmpty())
        return segment;
    double latitude = position[1].toDouble();
    double longitude = position[0].toDouble();
    QGeoCoordinate coord(latitude, longitude);

    QString geometry = step.value(QLatin1String("geometry")).toString();
    QList<QGeoCoordinate> path = decodePolyline(geometry);

    QGeoManeuver::InstructionDirection maneuverInstructionDirection = instructionDirection(maneuver, trafficSide);

    QString maneuverInstructionText = instructionText(step, maneuver, maneuverInstructionDirection);

    QGeoManeuver geoManeuver;
    geoManeuver.setDirection(maneuverInstructionDirection);
    geoManeuver.setDistanceToNextInstruction(distance);
    geoManeuver.setTimeToNextInstruction(time);
    geoManeuver.setInstructionText(maneuverInstructionText);
    geoManeuver.setPosition(coord);
    geoManeuver.setWaypoint(coord);

    QVariantMap extraAttributes;
    static const QStringList extras{
        QLatin1String("bearing_before"),
        QLatin1String("bearing_after"),
        QLatin1String("instruction"),
        QLatin1String("type"),
        QLatin1String("modifier")};
    for (const QString &e : extras)
    {
        if (maneuver.find(e) != maneuver.end())
            extraAttributes.insert(e, maneuver.value(e).toVariant());
    }
    // These should be removed as soon as route leg support is introduced.
    // Ref: http://project-osrm.org/docs/v5.15.2/api/#routeleg-object
    extraAttributes.insert(QLatin1String("leg_index"), legIndex);
    extraAttributes.insert(QLatin1String("step_index"), stepIndex);

    geoManeuver.setExtendedAttributes(extraAttributes);

    segment.setDistance(distance);
    segment.setPath(path);
    segment.setTravelTime(time);
    segment.setManeuver(geoManeuver);
    this->updateSegment(segment, step, maneuver);
    return segment;
}

QGeoRouteReply::Error MapboxRouteParser::parseReply(QList<QGeoRoute> &routes, QString &errorString, const QByteArray &reply) const
{
    // OSRM v5 specs: https://github.com/Project-OSRM/osrm-backend/blob/master/docs/http.md
    // Mapbox Directions API spec: https://www.mapbox.com/api-documentation/#directions
    QJsonDocument document = QJsonDocument::fromJson(reply);
    if (document.isObject())
    {
        QJsonObject object = document.object();

        QString status = object.value(QLatin1String("code")).toString();
        qWarning() << "status: " << status;
        if (status != QLatin1String("Ok"))
        {
            errorString = status;
            return QGeoRouteReply::UnknownError;
        }
        if (!object.value(QLatin1String("routes")).isArray())
        {
            errorString = QLatin1String("No routes found");
            return QGeoRouteReply::ParseError;
        }

        QJsonArray osrmRoutes = object.value(QLatin1String("routes")).toArray();
        foreach (const QJsonValue &r, osrmRoutes)
        {
            if (!r.isObject())
                continue;
            QJsonObject routeObject = r.toObject();
            if (!routeObject.value(QLatin1String("legs")).isArray())
                continue;
            if (!routeObject.value(QLatin1String("duration")).isDouble())
                continue;
            if (!routeObject.value(QLatin1String("distance")).isDouble())
                continue;

            double distance = routeObject.value(QLatin1String("distance")).toDouble();
            double travelTime = routeObject.value(QLatin1String("duration")).toDouble();
            qWarning() << "distance: " << distance;
            qWarning() << "Travel time: " << travelTime;
            bool error = false;
            QList<QGeoRouteSegment> segments;

            QJsonArray legs = routeObject.value(QLatin1String("legs")).toArray();
            QList<QGeoRouteLeg> routeLegs;
            QGeoRoute route;
            for (int legIndex = 0; legIndex < legs.size(); ++legIndex)
            {
                const QJsonValue &l = legs.at(legIndex);
                QGeoRouteLeg routeLeg;
                QList<QGeoRouteSegment> legSegments;
                if (!l.isObject())
                { // invalid leg record
                    error = true;
                    break;
                }
                QJsonObject leg = l.toObject();
                if (!leg.value(QLatin1String("steps")).isArray())
                { // Invalid steps field
                    error = true;
                    break;
                }
                const double legDistance = leg.value(QLatin1String("distance")).toDouble();
                const double legTravelTime = leg.value(QLatin1String("duration")).toDouble();
                QJsonArray steps = leg.value(QLatin1String("steps")).toArray();
                MapboxRouteSegment segment;
                for (int stepIndex = 0; stepIndex < steps.size(); ++stepIndex)
                {
                    const QJsonValue &s = steps.at(stepIndex);
                    if (!s.isObject())
                    {
                        error = true;
                        break;
                    }
                    segment = parseStep(s.toObject(), legIndex, stepIndex);
                    if (segment.isValid())
                    {
                        // setNextRouteSegment done below for all segments in the route.
                        legSegments.append(segment);
                    }
                    else
                    {
                        error = true;
                        break;
                    }
                }
                if (error)
                    break;

                segment.setLegLastSegment(true);

                QList<QGeoCoordinate> path;
                for (const QGeoRouteSegment &s : qAsConst(legSegments))
                    path.append(s.path());
                routeLeg.setLegIndex(legIndex);
                routeLeg.setOverallRoute(route); // QGeoRoute::d_ptr is explicitlySharedDataPointer. Modifiers below won't detach it.
                routeLeg.setDistance(legDistance);
                routeLeg.setTravelTime(legTravelTime);
                if (!path.isEmpty())
                {
                    routeLeg.setPath(path);
                    routeLeg.setFirstRouteSegment(legSegments.first());
                }
                routeLegs << routeLeg;

                segments.append(legSegments);
            }

            if (!error)
            {
                QList<QGeoCoordinate> path;
                for (const QGeoRouteSegment &s : segments)
                    path.append(s.path());

                for (int i = segments.size() - 1; i > 0; --i)
                    segments[i - 1].setNextRouteSegment(segments[i]);

                qWarning() << "distance: " << distance;
                route.setDistance(distance);
                qWarning() << "Travel time: " << travelTime;
                route.setTravelTime(travelTime);
                if (!path.isEmpty())
                {
                    qWarning() << "Path: " << path;
                    route.setPath(path);
                    route.setBounds(QGeoPath(path).boundingGeoRectangle());
                    qWarning() << "First segment: " << segments.first().distance();
                    route.setFirstRouteSegment(segments.first());
                }
                route.setRouteLegs(routeLegs);
                //r.setTravelMode(QGeoRouteRequest::CarTravel); // The only one supported by OSRM demo service, but other OSRM servers might do cycle or pedestrian too
                routes.append(route);
            }
        }

        // setError(QGeoRouteReply::NoError, status);  // can't do this, or NoError is emitted and does damages
        return QGeoRouteReply::NoError;
    }
    else
    {
        errorString = QLatin1String("Couldn't parse json.");
        return QGeoRouteReply::ParseError;
    }
}

QUrl MapboxRouteParser::requestUrl(const QGeoRouteRequest &request, const QString &prefix) const
{
    QString routingUrl = prefix;
    int notFirst = 0;
    QString bearings;
    const QList<QVariantMap> metadata = request.waypointsMetadata();
    const QList<QGeoCoordinate> waypoints = request.waypoints();
    for (int i = 0; i < waypoints.size(); i++)
    {
        const QGeoCoordinate &c = waypoints.at(i);
        if (notFirst)
        {
            routingUrl.append(QLatin1Char(';'));
            bearings.append(QLatin1Char(';'));
        }
        routingUrl.append(QString::number(c.longitude(), 'f', 7)).append(QLatin1Char(',')).append(QString::number(c.latitude(), 'f', 7));
        if (metadata.size() > i)
        {
            const QVariantMap &meta = metadata.at(i);
            if (meta.contains(QLatin1String("bearing")))
            {
                qreal bearing = meta.value(QLatin1String("bearing")).toDouble();
                bearings.append(QString::number(int(bearing))).append(QLatin1Char(',')).append(QLatin1String("90")); // 90 is the angle of maneuver allowed.
            }
            else
            {
                bearings.append(QLatin1String("0,180")); // 180 here means anywhere
            }
        }
        ++notFirst;
    }

    QUrl url(routingUrl);
    QUrlQuery query;
    query.addQueryItem(QLatin1String("overview"), QLatin1String("full"));
    query.addQueryItem(QLatin1String("steps"), QLatin1String("true"));
    query.addQueryItem(QLatin1String("geometries"), QLatin1String("polyline6"));
    query.addQueryItem(QLatin1String("alternatives"), QLatin1String("true"));
    query.addQueryItem(QLatin1String("bearings"), bearings);

    if (!m_accessToken.isEmpty())
        query.addQueryItem(QLatin1String("access_token"), m_accessToken);

    query.addQueryItem(QLatin1String("annotations"), QLatin1String("duration,distance,speed,congestion"));

    query.addQueryItem(QLatin1String("voice_instructions"), QLatin1String("true"));
    query.addQueryItem(QLatin1String("banner_instructions"), QLatin1String("true"));
    query.addQueryItem(QLatin1String("roundabout_exits"), QLatin1String("true"));

    query.addQueryItem(QLatin1String("voice_units"), Params().getBool("IsMetric") ? QLatin1String("metric") : QLatin1String("imperial"));

    url.setQuery(query);
    return url;
}

// void MapboxRouteParser::setTrafficSide(MapboxRouteParser::TrafficSide side)
// {
//     if (trafficSide != side)
//     {
//         trafficSide = side;
//         emit trafficSideChanged(trafficSide);
//     }
// }
