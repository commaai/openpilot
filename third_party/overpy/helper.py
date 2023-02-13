__author__ = 'mjob'

import overpy


def get_street(street, areacode, api=None):
    """
    Retrieve streets in a given bounding area

    :param overpy.Overpass api: First street of intersection
    :param String street: Name of street
    :param String areacode: The OSM id of the bounding area
    :return: Parsed result
    :raises overpy.exception.OverPyException: If something bad happens.
    """
    if api is None:
        api = overpy.Overpass()

    query = """
        area(%s)->.location;
        (
            way[highway][name="%s"](area.location);
            - (
                way[highway=service](area.location);
                way[highway=track](area.location);
            );
        );
        out body;
        >;
        out skel qt;
    """

    data = api.query(query % (areacode, street))

    return data


def get_intersection(street1, street2, areacode, api=None):
    """
    Retrieve intersection of two streets in a given bounding area

    :param overpy.Overpass api: First street of intersection
    :param String street1: Name of first street of intersection
    :param String street2: Name of second street of intersection
    :param String areacode: The OSM id of the bounding area
    :return: List of intersections
    :raises overpy.exception.OverPyException: If something bad happens.
    """
    if api is None:
        api = overpy.Overpass()

    query = """
        area(%s)->.location;
        (
            way[highway][name="%s"](area.location); node(w)->.n1;
            way[highway][name="%s"](area.location); node(w)->.n2;
        );
        node.n1.n2;
        out meta;
    """

    data = api.query(query % (areacode, street1, street2))

    return data.get_nodes()
