from collections import OrderedDict
from datetime import datetime
from decimal import Decimal
from xml.sax import handler, make_parser
import json
import re
import sys

from overpy import exception
from overpy.__about__ import (
    __author__, __copyright__, __email__, __license__, __summary__, __title__,
    __uri__, __version__
)

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

XML_PARSER_DOM = 1
XML_PARSER_SAX = 2

if PY2:
    from urllib2 import urlopen
    from urllib2 import HTTPError
elif PY3:
    from urllib.request import urlopen
    from urllib.error import HTTPError


def is_valid_type(element, cls):
    """
    Test if an element is of a given type.

    :param Element() element: The element instance to test
    :param Element cls: The element class to test
    :return: False or True
    :rtype: Boolean
    """
    return isinstance(element, cls) and element.id is not None


class Overpass(object):
    """
    Class to access the Overpass API
    """
    default_read_chunk_size = 4096
    default_url = "http://overpass-api.de/api/interpreter"

    def __init__(self, read_chunk_size=None, url=None, xml_parser=XML_PARSER_SAX):
        """
        :param read_chunk_size: Max size of each chunk read from the server response
        :type read_chunk_size: Integer
        :param url: Optional URL of the Overpass server. Defaults to http://overpass-api.de/api/interpreter
        :type url: str
        :param xml_parser: The xml parser to use
        :type xml_parser: Integer
        """
        self.url = self.default_url
        if url is not None:
            self.url = url

        self._regex_extract_error_msg = re.compile(b"\<p\>(?P<msg>\<strong\s.*?)\</p\>")
        self._regex_remove_tag = re.compile(b"<[^>]*?>")
        if read_chunk_size is None:
            read_chunk_size = self.default_read_chunk_size
        self.read_chunk_size = read_chunk_size
        self.xml_parser = xml_parser

    def query(self, query):
        """
        Query the Overpass API

        :param String|Bytes query: The query string in Overpass QL
        :return: The parsed result
        :rtype: overpy.Result
        """
        if not isinstance(query, bytes):
            query = query.encode("utf-8")

        try:
            f = urlopen(self.url, query)
        except HTTPError as e:
            f = e

        response = f.read(self.read_chunk_size)
        while True:
            data = f.read(self.read_chunk_size)
            if len(data) == 0:
                break
            response = response + data
        f.close()

        if f.code == 200:
            if PY2:
                http_info = f.info()
                content_type = http_info.getheader("content-type")
            else:
                content_type = f.getheader("Content-Type")

            if content_type == "application/json":
                return self.parse_json(response)

            if content_type == "application/osm3s+xml":
                return self.parse_xml(response)

            raise exception.OverpassUnknownContentType(content_type)

        if f.code == 400:
            msgs = []
            for msg in self._regex_extract_error_msg.finditer(response):
                tmp = self._regex_remove_tag.sub(b"", msg.group("msg"))
                try:
                    tmp = tmp.decode("utf-8")
                except UnicodeDecodeError:
                    tmp = repr(tmp)
                msgs.append(tmp)

            raise exception.OverpassBadRequest(
                query,
                msgs=msgs
            )

        if f.code == 429:
            raise exception.OverpassTooManyRequests

        if f.code == 504:
            raise exception.OverpassGatewayTimeout

        raise exception.OverpassUnknownHTTPStatusCode(f.code)

    def parse_json(self, data, encoding="utf-8"):
        """
        Parse raw response from Overpass service.

        :param data: Raw JSON Data
        :type data: String or Bytes
        :param encoding: Encoding to decode byte string
        :type encoding: String
        :return: Result object
        :rtype: overpy.Result
        """
        if isinstance(data, bytes):
            data = data.decode(encoding)
        data = json.loads(data, parse_float=Decimal)
        return Result.from_json(data, api=self)

    def parse_xml(self, data, encoding="utf-8", parser=None):
        """

        :param data: Raw XML Data
        :type data: String or Bytes
        :param encoding: Encoding to decode byte string
        :type encoding: String
        :return: Result object
        :rtype: overpy.Result
        """
        if parser is None:
            parser = self.xml_parser

        if isinstance(data, bytes):
            data = data.decode(encoding)
        if PY2 and not isinstance(data, str):
            # Python 2.x: Convert unicode strings
            data = data.encode(encoding)

        return Result.from_xml(data, api=self, parser=parser)


class Result(object):
    """
    Class to handle the result.
    """

    def __init__(self, elements=None, api=None):
        """

        :param List elements:
        :param api:
        :type api: overpy.Overpass
        """
        if elements is None:
            elements = []
        self._areas = OrderedDict((element.id, element) for element in elements if is_valid_type(element, Area))
        self._nodes = OrderedDict((element.id, element) for element in elements if is_valid_type(element, Node))
        self._ways = OrderedDict((element.id, element) for element in elements if is_valid_type(element, Way))
        self._relations = OrderedDict((element.id, element)
                                      for element in elements if is_valid_type(element, Relation))
        self._class_collection_map = {Node: self._nodes, Way: self._ways, Relation: self._relations, Area: self._areas}
        self.api = api

    def expand(self, other):
        """
        Add all elements from an other result to the list of elements of this result object.

        It is used by the auto resolve feature.

        :param other: Expand the result with the elements from this result.
        :type other: overpy.Result
        :raises ValueError: If provided parameter is not instance of :class:`overpy.Result`
        """
        if not isinstance(other, Result):
            raise ValueError("Provided argument has to be instance of overpy:Result()")

        other_collection_map = {Node: other.nodes, Way: other.ways, Relation: other.relations, Area: other.areas}
        for element_type, own_collection in self._class_collection_map.items():
            for element in other_collection_map[element_type]:
                if is_valid_type(element, element_type) and element.id not in own_collection:
                    own_collection[element.id] = element

    def append(self, element):
        """
        Append a new element to the result.

        :param element: The element to append
        :type element: overpy.Element
        """
        if is_valid_type(element, Element):
            self._class_collection_map[element.__class__].setdefault(element.id, element)

    def get_elements(self, filter_cls, elem_id=None):
        """
        Get a list of elements from the result and filter the element type by a class.

        :param filter_cls:
        :param elem_id: ID of the object
        :type elem_id: Integer
        :return: List of available elements
        :rtype: List
        """
        result = []
        if elem_id is not None:
            try:
                result = [self._class_collection_map[filter_cls][elem_id]]
            except KeyError:
                result = []
        else:
            for e in self._class_collection_map[filter_cls].values():
                result.append(e)
        return result

    def get_ids(self, filter_cls):
        """

        :param filter_cls:
        :return:
        """
        return list(self._class_collection_map[filter_cls].keys())

    def get_node_ids(self):
        return self.get_ids(filter_cls=Node)

    def get_way_ids(self):
        return self.get_ids(filter_cls=Way)

    def get_relation_ids(self):
        return self.get_ids(filter_cls=Relation)

    def get_area_ids(self):
        return self.get_ids(filter_cls=Area)

    @classmethod
    def from_json(cls, data, api=None):
        """
        Create a new instance and load data from json object.

        :param data: JSON data returned by the Overpass API
        :type data: Dict
        :param api:
        :type api: overpy.Overpass
        :return: New instance of Result object
        :rtype: overpy.Result
        """
        result = cls(api=api)
        for elem_cls in [Node, Way, Relation, Area]:
            for element in data.get("elements", []):
                e_type = element.get("type")
                if hasattr(e_type, "lower") and e_type.lower() == elem_cls._type_value:
                    result.append(elem_cls.from_json(element, result=result))

        return result

    @classmethod
    def from_xml(cls, data, api=None, parser=XML_PARSER_SAX):
        """
        Create a new instance and load data from xml object.

        :param data: Root element
        :type data: xml.etree.ElementTree.Element
        :param api:
        :type api: Overpass
        :param parser: Specify the parser to use(DOM or SAX)
        :type parser: Integer
        :return: New instance of Result object
        :rtype: Result
        """
        result = cls(api=api)
        if parser == XML_PARSER_DOM:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(data)

            for elem_cls in [Node, Way, Relation, Area]:
                for child in root:
                    if child.tag.lower() == elem_cls._type_value:
                        result.append(elem_cls.from_xml(child, result=result))

        elif parser == XML_PARSER_SAX:
            if PY2:
                from StringIO import StringIO
            else:
                from io import StringIO
            source = StringIO(data)
            sax_handler = OSMSAXHandler(result)
            parser = make_parser()
            parser.setContentHandler(sax_handler)
            parser.parse(source)
        else:
            # ToDo: better exception
            raise Exception("Unknown XML parser")
        return result

    def get_area(self, area_id, resolve_missing=False):
        """
        Get an area by its ID.

        :param area_id: The area ID
        :type area_id: Integer
        :param resolve_missing: Query the Overpass API if the area is missing in the result set.
        :return: The area
        :rtype: overpy.Area
        :raises overpy.exception.DataIncomplete: The requested way is not available in the result cache.
        :raises overpy.exception.DataIncomplete: If resolve_missing is True and the area can't be resolved.
        """
        areas = self.get_areas(area_id=area_id)
        if len(areas) == 0:
            if resolve_missing is False:
                raise exception.DataIncomplete("Resolve missing area is disabled")

            query = ("\n"
                     "[out:json];\n"
                     "area({area_id});\n"
                     "out body;\n"
                     )
            query = query.format(
                area_id=area_id
            )
            tmp_result = self.api.query(query)
            self.expand(tmp_result)

            areas = self.get_areas(area_id=area_id)

        if len(areas) == 0:
            raise exception.DataIncomplete("Unable to resolve requested areas")

        return areas[0]

    def get_areas(self, area_id=None, **kwargs):
        """
        Alias for get_elements() but filter the result by Area

        :param area_id: The Id of the area
        :type area_id: Integer
        :return: List of elements
        """
        return self.get_elements(Area, elem_id=area_id, **kwargs)

    def get_node(self, node_id, resolve_missing=False):
        """
        Get a node by its ID.

        :param node_id: The node ID
        :type node_id: Integer
        :param resolve_missing: Query the Overpass API if the node is missing in the result set.
        :return: The node
        :rtype: overpy.Node
        :raises overpy.exception.DataIncomplete: At least one referenced node is not available in the result cache.
        :raises overpy.exception.DataIncomplete: If resolve_missing is True and at least one node can't be resolved.
        """
        nodes = self.get_nodes(node_id=node_id)
        if len(nodes) == 0:
            if not resolve_missing:
                raise exception.DataIncomplete("Resolve missing nodes is disabled")

            query = ("\n"
                     "[out:json];\n"
                     "node({node_id});\n"
                     "out body;\n"
                     )
            query = query.format(
                node_id=node_id
            )
            tmp_result = self.api.query(query)
            self.expand(tmp_result)

            nodes = self.get_nodes(node_id=node_id)

        if len(nodes) == 0:
            raise exception.DataIncomplete("Unable to resolve all nodes")

        return nodes[0]

    def get_nodes(self, node_id=None, **kwargs):
        """
        Alias for get_elements() but filter the result by Node()

        :param node_id: The Id of the node
        :type node_id: Integer
        :return: List of elements
        """
        return self.get_elements(Node, elem_id=node_id, **kwargs)

    def get_relation(self, rel_id, resolve_missing=False):
        """
        Get a relation by its ID.

        :param rel_id: The relation ID
        :type rel_id: Integer
        :param resolve_missing: Query the Overpass API if the relation is missing in the result set.
        :return: The relation
        :rtype: overpy.Relation
        :raises overpy.exception.DataIncomplete: The requested relation is not available in the result cache.
        :raises overpy.exception.DataIncomplete: If resolve_missing is True and the relation can't be resolved.
        """
        relations = self.get_relations(rel_id=rel_id)
        if len(relations) == 0:
            if resolve_missing is False:
                raise exception.DataIncomplete("Resolve missing relations is disabled")

            query = ("\n"
                     "[out:json];\n"
                     "relation({relation_id});\n"
                     "out body;\n"
                     )
            query = query.format(
                relation_id=rel_id
            )
            tmp_result = self.api.query(query)
            self.expand(tmp_result)

            relations = self.get_relations(rel_id=rel_id)

        if len(relations) == 0:
            raise exception.DataIncomplete("Unable to resolve requested reference")

        return relations[0]

    def get_relations(self, rel_id=None, **kwargs):
        """
        Alias for get_elements() but filter the result by Relation

        :param rel_id: Id of the relation
        :type rel_id: Integer
        :return: List of elements
        """
        return self.get_elements(Relation, elem_id=rel_id, **kwargs)

    def get_way(self, way_id, resolve_missing=False):
        """
        Get a way by its ID.

        :param way_id: The way ID
        :type way_id: Integer
        :param resolve_missing: Query the Overpass API if the way is missing in the result set.
        :return: The way
        :rtype: overpy.Way
        :raises overpy.exception.DataIncomplete: The requested way is not available in the result cache.
        :raises overpy.exception.DataIncomplete: If resolve_missing is True and the way can't be resolved.
        """
        ways = self.get_ways(way_id=way_id)
        if len(ways) == 0:
            if resolve_missing is False:
                raise exception.DataIncomplete("Resolve missing way is disabled")

            query = ("\n"
                     "[out:json];\n"
                     "way({way_id});\n"
                     "out body;\n"
                     )
            query = query.format(
                way_id=way_id
            )
            tmp_result = self.api.query(query)
            self.expand(tmp_result)

            ways = self.get_ways(way_id=way_id)

        if len(ways) == 0:
            raise exception.DataIncomplete("Unable to resolve requested way")

        return ways[0]

    def get_ways(self, way_id=None, **kwargs):
        """
        Alias for get_elements() but filter the result by Way

        :param way_id: The Id of the way
        :type way_id: Integer
        :return: List of elements
        """
        return self.get_elements(Way, elem_id=way_id, **kwargs)

    area_ids = property(get_area_ids)
    areas = property(get_areas)
    node_ids = property(get_node_ids)
    nodes = property(get_nodes)
    relation_ids = property(get_relation_ids)
    relations = property(get_relations)
    way_ids = property(get_way_ids)
    ways = property(get_ways)


class Element(object):
    """
    Base element
    """

    def __init__(self, attributes=None, result=None, tags=None):
        """
        :param attributes: Additional attributes
        :type attributes: Dict
        :param result: The result object this element belongs to
        :param tags: List of tags
        :type tags: Dict
        """

        self._result = result
        # Try to convert some common attributes
        # http://wiki.openstreetmap.org/wiki/Elements#Common_attributes
        self._attribute_modifiers = {
            "changeset": int,
            "timestamp": lambda ts: datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"),
            "uid": int,
            "version": int,
            "visible": lambda v: v.lower() == "true"
        }
        self.attributes = attributes
        for n, m in self._attribute_modifiers.items():
            if n in self.attributes:
                self.attributes[n] = m(self.attributes[n])
        self.id = None
        self.tags = tags

    @classmethod
    def get_center_from_json(cls, data):
        """
        Get center information from json data

        :param data: json data
        :return: tuple with two elements: lat and lon
        :rtype: tuple
        """
        center_lat = None
        center_lon = None
        center = data.get("center")
        if isinstance(center, dict):
            center_lat = center.get("lat")
            center_lon = center.get("lon")
            if center_lat is None or center_lon is None:
                raise ValueError("Unable to get lat or lon of way center.")
            center_lat = Decimal(center_lat)
            center_lon = Decimal(center_lon)
        return (center_lat, center_lon)

    @classmethod
    def get_center_from_xml_dom(cls, sub_child):
        center_lat = sub_child.attrib.get("lat")
        center_lon = sub_child.attrib.get("lon")
        if center_lat is None or center_lon is None:
            raise ValueError("Unable to get lat or lon of way center.")
        center_lat = Decimal(center_lat)
        center_lon = Decimal(center_lon)
        return center_lat, center_lon


class Area(Element):
    """
    Class to represent an element of type area
    """

    _type_value = "area"

    def __init__(self, area_id=None, **kwargs):
        """
        :param area_id: Id of the area element
        :type area_id: Integer
        :param kwargs: Additional arguments are passed directly to the parent class

        """

        Element.__init__(self, **kwargs)
        #: The id of the way
        self.id = area_id

    def __repr__(self):
        return "<overpy.Area id={}>".format(self.id)

    @classmethod
    def from_json(cls, data, result=None):
        """
        Create new Area element from JSON data

        :param data: Element data from JSON
        :type data: Dict
        :param result: The result this element belongs to
        :type result: overpy.Result
        :return: New instance of Way
        :rtype: overpy.Area
        :raises overpy.exception.ElementDataWrongType: If type value of the passed JSON data does not match.
        """
        if data.get("type") != cls._type_value:
            raise exception.ElementDataWrongType(
                type_expected=cls._type_value,
                type_provided=data.get("type")
            )

        tags = data.get("tags", {})

        area_id = data.get("id")

        attributes = {}
        ignore = ["id", "tags", "type"]
        for n, v in data.items():
            if n in ignore:
                continue
            attributes[n] = v

        return cls(area_id=area_id, attributes=attributes, tags=tags, result=result)

    @classmethod
    def from_xml(cls, child, result=None):
        """
        Create new way element from XML data

        :param child: XML node to be parsed
        :type child: xml.etree.ElementTree.Element
        :param result: The result this node belongs to
        :type result: overpy.Result
        :return: New Way oject
        :rtype: overpy.Way
        :raises overpy.exception.ElementDataWrongType: If name of the xml child node doesn't match
        :raises ValueError: If the ref attribute of the xml node is not provided
        :raises ValueError: If a tag doesn't have a name
        """
        if child.tag.lower() != cls._type_value:
            raise exception.ElementDataWrongType(
                type_expected=cls._type_value,
                type_provided=child.tag.lower()
            )

        tags = {}

        for sub_child in child:
            if sub_child.tag.lower() == "tag":
                name = sub_child.attrib.get("k")
                if name is None:
                    raise ValueError("Tag without name/key.")
                value = sub_child.attrib.get("v")
                tags[name] = value

        area_id = child.attrib.get("id")
        if area_id is not None:
            area_id = int(area_id)

        attributes = {}
        ignore = ["id"]
        for n, v in child.attrib.items():
            if n in ignore:
                continue
            attributes[n] = v

        return cls(area_id=area_id, attributes=attributes, tags=tags, result=result)


class Node(Element):
    """
    Class to represent an element of type node
    """

    _type_value = "node"

    def __init__(self, node_id=None, lat=None, lon=None, **kwargs):
        """
        :param lat: Latitude
        :type lat: Decimal or Float
        :param lon: Longitude
        :type long: Decimal or Float
        :param node_id: Id of the node element
        :type node_id: Integer
        :param kwargs: Additional arguments are passed directly to the parent class
        """

        Element.__init__(self, **kwargs)
        self.id = node_id
        self.lat = lat
        self.lon = lon

    def __repr__(self):
        return "<overpy.Node id={} lat={} lon={}>".format(self.id, self.lat, self.lon)

    @classmethod
    def from_json(cls, data, result=None):
        """
        Create new Node element from JSON data

        :param data: Element data from JSON
        :type data: Dict
        :param result: The result this element belongs to
        :type result: overpy.Result
        :return: New instance of Node
        :rtype: overpy.Node
        :raises overpy.exception.ElementDataWrongType: If type value of the passed JSON data does not match.
        """
        if data.get("type") != cls._type_value:
            raise exception.ElementDataWrongType(
                type_expected=cls._type_value,
                type_provided=data.get("type")
            )

        tags = data.get("tags", {})

        node_id = data.get("id")
        lat = data.get("lat")
        lon = data.get("lon")

        attributes = {}
        ignore = ["type", "id", "lat", "lon", "tags"]
        for n, v in data.items():
            if n in ignore:
                continue
            attributes[n] = v

        return cls(node_id=node_id, lat=lat, lon=lon, tags=tags, attributes=attributes, result=result)

    @classmethod
    def from_xml(cls, child, result=None):
        """
        Create new way element from XML data

        :param child: XML node to be parsed
        :type child: xml.etree.ElementTree.Element
        :param result: The result this node belongs to
        :type result: overpy.Result
        :return: New Way oject
        :rtype: overpy.Node
        :raises overpy.exception.ElementDataWrongType: If name of the xml child node doesn't match
        :raises ValueError: If a tag doesn't have a name
        """
        if child.tag.lower() != cls._type_value:
            raise exception.ElementDataWrongType(
                type_expected=cls._type_value,
                type_provided=child.tag.lower()
            )

        tags = {}

        for sub_child in child:
            if sub_child.tag.lower() == "tag":
                name = sub_child.attrib.get("k")
                if name is None:
                    raise ValueError("Tag without name/key.")
                value = sub_child.attrib.get("v")
                tags[name] = value

        node_id = child.attrib.get("id")
        if node_id is not None:
            node_id = int(node_id)
        lat = child.attrib.get("lat")
        if lat is not None:
            lat = Decimal(lat)
        lon = child.attrib.get("lon")
        if lon is not None:
            lon = Decimal(lon)

        attributes = {}
        ignore = ["id", "lat", "lon"]
        for n, v in child.attrib.items():
            if n in ignore:
                continue
            attributes[n] = v

        return cls(node_id=node_id, lat=lat, lon=lon, tags=tags, attributes=attributes, result=result)


class Way(Element):
    """
    Class to represent an element of type way
    """

    _type_value = "way"

    def __init__(self, way_id=None, center_lat=None, center_lon=None, node_ids=None, **kwargs):
        """
        :param node_ids: List of node IDs
        :type node_ids: List or Tuple
        :param way_id: Id of the way element
        :type way_id: Integer
        :param kwargs: Additional arguments are passed directly to the parent class

        """

        Element.__init__(self, **kwargs)
        #: The id of the way
        self.id = way_id

        #: List of Ids of the associated nodes
        self._node_ids = node_ids

        #: The lat/lon of the center of the way (optional depending on query)
        self.center_lat = center_lat
        self.center_lon = center_lon

    def __repr__(self):
        return "<overpy.Way id={} nodes={}>".format(self.id, self._node_ids)

    @property
    def nodes(self):
        """
        List of nodes associated with the way.
        """
        return self.get_nodes()

    def get_nodes(self, resolve_missing=False):
        """
        Get the nodes defining the geometry of the way

        :param resolve_missing: Try to resolve missing nodes.
        :type resolve_missing: Boolean
        :return: List of nodes
        :rtype: List of overpy.Node
        :raises overpy.exception.DataIncomplete: At least one referenced node is not available in the result cache.
        :raises overpy.exception.DataIncomplete: If resolve_missing is True and at least one node can't be resolved.
        """
        result = []
        resolved = False

        for node_id in self._node_ids:
            try:
                node = self._result.get_node(node_id)
            except exception.DataIncomplete:
                node = None

            if node is not None:
                result.append(node)
                continue

            if not resolve_missing:
                raise exception.DataIncomplete("Resolve missing nodes is disabled")

            # We tried to resolve the data but some nodes are still missing
            if resolved:
                raise exception.DataIncomplete("Unable to resolve all nodes")

            query = ("\n"
                     "[out:json];\n"
                     "way({way_id});\n"
                     "node(w);\n"
                     "out body;\n"
                     )
            query = query.format(
                way_id=self.id
            )
            tmp_result = self._result.api.query(query)
            self._result.expand(tmp_result)
            resolved = True

            try:
                node = self._result.get_node(node_id)
            except exception.DataIncomplete:
                node = None

            if node is None:
                raise exception.DataIncomplete("Unable to resolve all nodes")

            result.append(node)

        return result

    @classmethod
    def from_json(cls, data, result=None):
        """
        Create new Way element from JSON data

        :param data: Element data from JSON
        :type data: Dict
        :param result: The result this element belongs to
        :type result: overpy.Result
        :return: New instance of Way
        :rtype: overpy.Way
        :raises overpy.exception.ElementDataWrongType: If type value of the passed JSON data does not match.
        """
        if data.get("type") != cls._type_value:
            raise exception.ElementDataWrongType(
                type_expected=cls._type_value,
                type_provided=data.get("type")
            )

        tags = data.get("tags", {})

        way_id = data.get("id")
        node_ids = data.get("nodes")
        (center_lat, center_lon) = cls.get_center_from_json(data=data)

        attributes = {}
        ignore = ["center", "id", "nodes", "tags", "type"]
        for n, v in data.items():
            if n in ignore:
                continue
            attributes[n] = v

        return cls(
            attributes=attributes,
            center_lat=center_lat,
            center_lon=center_lon,
            node_ids=node_ids,
            tags=tags,
            result=result,
            way_id=way_id
        )

    @classmethod
    def from_xml(cls, child, result=None):
        """
        Create new way element from XML data

        :param child: XML node to be parsed
        :type child: xml.etree.ElementTree.Element
        :param result: The result this node belongs to
        :type result: overpy.Result
        :return: New Way oject
        :rtype: overpy.Way
        :raises overpy.exception.ElementDataWrongType: If name of the xml child node doesn't match
        :raises ValueError: If the ref attribute of the xml node is not provided
        :raises ValueError: If a tag doesn't have a name
        """
        if child.tag.lower() != cls._type_value:
            raise exception.ElementDataWrongType(
                type_expected=cls._type_value,
                type_provided=child.tag.lower()
            )

        tags = {}
        node_ids = []
        center_lat = None
        center_lon = None

        for sub_child in child:
            if sub_child.tag.lower() == "tag":
                name = sub_child.attrib.get("k")
                if name is None:
                    raise ValueError("Tag without name/key.")
                value = sub_child.attrib.get("v")
                tags[name] = value
            if sub_child.tag.lower() == "nd":
                ref_id = sub_child.attrib.get("ref")
                if ref_id is None:
                    raise ValueError("Unable to find required ref value.")
                ref_id = int(ref_id)
                node_ids.append(ref_id)
            if sub_child.tag.lower() == "center":
                (center_lat, center_lon) = cls.get_center_from_xml_dom(sub_child=sub_child)

        way_id = child.attrib.get("id")
        if way_id is not None:
            way_id = int(way_id)

        attributes = {}
        ignore = ["id"]
        for n, v in child.attrib.items():
            if n in ignore:
                continue
            attributes[n] = v

        return cls(way_id=way_id, center_lat=center_lat, center_lon=center_lon,
                   attributes=attributes, node_ids=node_ids, tags=tags, result=result)


class Relation(Element):
    """
    Class to represent an element of type relation
    """

    _type_value = "relation"

    def __init__(self, rel_id=None, center_lat=None, center_lon=None, members=None, **kwargs):
        """
        :param members:
        :param rel_id: Id of the relation element
        :type rel_id: Integer
        :param kwargs:
        :return:
        """

        Element.__init__(self, **kwargs)
        self.id = rel_id
        self.members = members

        #: The lat/lon of the center of the way (optional depending on query)
        self.center_lat = center_lat
        self.center_lon = center_lon

    def __repr__(self):
        return "<overpy.Relation id={}>".format(self.id)

    @classmethod
    def from_json(cls, data, result=None):
        """
        Create new Relation element from JSON data

        :param data: Element data from JSON
        :type data: Dict
        :param result: The result this element belongs to
        :type result: overpy.Result
        :return: New instance of Relation
        :rtype: overpy.Relation
        :raises overpy.exception.ElementDataWrongType: If type value of the passed JSON data does not match.
        """
        if data.get("type") != cls._type_value:
            raise exception.ElementDataWrongType(
                type_expected=cls._type_value,
                type_provided=data.get("type")
            )

        tags = data.get("tags", {})

        rel_id = data.get("id")
        (center_lat, center_lon) = cls.get_center_from_json(data=data)

        members = []

        supported_members = [RelationNode, RelationWay, RelationRelation]
        for member in data.get("members", []):
            type_value = member.get("type")
            for member_cls in supported_members:
                if member_cls._type_value == type_value:
                    members.append(
                        member_cls.from_json(
                            member,
                            result=result
                        )
                    )

        attributes = {}
        ignore = ["id", "members", "tags", "type"]
        for n, v in data.items():
            if n in ignore:
                continue
            attributes[n] = v

        return cls(
            rel_id=rel_id,
            attributes=attributes,
            center_lat=center_lat,
            center_lon=center_lon,
            members=members,
            tags=tags,
            result=result
        )

    @classmethod
    def from_xml(cls, child, result=None):
        """
        Create new way element from XML data

        :param child: XML node to be parsed
        :type child: xml.etree.ElementTree.Element
        :param result: The result this node belongs to
        :type result: overpy.Result
        :return: New Way oject
        :rtype: overpy.Relation
        :raises overpy.exception.ElementDataWrongType: If name of the xml child node doesn't match
        :raises ValueError: If a tag doesn't have a name
        """
        if child.tag.lower() != cls._type_value:
            raise exception.ElementDataWrongType(
                type_expected=cls._type_value,
                type_provided=child.tag.lower()
            )

        tags = {}
        members = []
        center_lat = None
        center_lon = None

        supported_members = [RelationNode, RelationWay, RelationRelation, RelationArea]
        for sub_child in child:
            if sub_child.tag.lower() == "tag":
                name = sub_child.attrib.get("k")
                if name is None:
                    raise ValueError("Tag without name/key.")
                value = sub_child.attrib.get("v")
                tags[name] = value
            if sub_child.tag.lower() == "member":
                type_value = sub_child.attrib.get("type")
                for member_cls in supported_members:
                    if member_cls._type_value == type_value:
                        members.append(
                            member_cls.from_xml(
                                sub_child,
                                result=result
                            )
                        )
            if sub_child.tag.lower() == "center":
                (center_lat, center_lon) = cls.get_center_from_xml_dom(sub_child=sub_child)

        rel_id = child.attrib.get("id")
        if rel_id is not None:
            rel_id = int(rel_id)

        attributes = {}
        ignore = ["id"]
        for n, v in child.attrib.items():
            if n in ignore:
                continue
            attributes[n] = v

        return cls(
            rel_id=rel_id,
            attributes=attributes,
            center_lat=center_lat,
            center_lon=center_lon,
            members=members,
            tags=tags,
            result=result
        )


class RelationMember(object):
    """
    Base class to represent a member of a relation.
    """

    def __init__(self, attributes=None, geometry=None, ref=None, role=None, result=None):
        """
        :param ref: Reference Id
        :type ref: Integer
        :param role: The role of the relation member
        :type role: String
        :param result:
        """
        self.ref = ref
        self._result = result
        self.role = role
        self.attributes = attributes
        self.geometry = geometry

    @classmethod
    def from_json(cls, data, result=None):
        """
        Create new RelationMember element from JSON data

        :param child: Element data from JSON
        :type child: Dict
        :param result: The result this element belongs to
        :type result: overpy.Result
        :return: New instance of RelationMember
        :rtype: overpy.RelationMember
        :raises overpy.exception.ElementDataWrongType: If type value of the passed JSON data does not match.
        """
        if data.get("type") != cls._type_value:
            raise exception.ElementDataWrongType(
                type_expected=cls._type_value,
                type_provided=data.get("type")
            )

        ref = data.get("ref")
        role = data.get("role")

        attributes = {}
        ignore = ["geometry", "type", "ref", "role"]
        for n, v in data.items():
            if n in ignore:
                continue
            attributes[n] = v

        geometry = data.get("geometry")
        if isinstance(geometry, list):
            geometry_orig = geometry
            geometry = []
            for v in geometry_orig:
                geometry.append(
                    RelationWayGeometryValue(
                        lat=v.get("lat"),
                        lon=v.get("lon")
                    )
                )
        else:
            geometry = None

        return cls(
            attributes=attributes,
            geometry=geometry,
            ref=ref,
            role=role,
            result=result
        )

    @classmethod
    def from_xml(cls, child, result=None):
        """
        Create new RelationMember from XML data

        :param child: XML node to be parsed
        :type child: xml.etree.ElementTree.Element
        :param result: The result this element belongs to
        :type result: overpy.Result
        :return: New relation member oject
        :rtype: overpy.RelationMember
        :raises overpy.exception.ElementDataWrongType: If name of the xml child node doesn't match
        """
        if child.attrib.get("type") != cls._type_value:
            raise exception.ElementDataWrongType(
                type_expected=cls._type_value,
                type_provided=child.tag.lower()
            )

        ref = child.attrib.get("ref")
        if ref is not None:
            ref = int(ref)
        role = child.attrib.get("role")

        attributes = {}
        ignore = ["geometry", "ref", "role", "type"]
        for n, v in child.attrib.items():
            if n in ignore:
                continue
            attributes[n] = v

        geometry = None
        for sub_child in child:
            if sub_child.tag.lower() == "nd":
                if geometry is None:
                    geometry = []
                geometry.append(
                    RelationWayGeometryValue(
                        lat=Decimal(sub_child.attrib["lat"]),
                        lon=Decimal(sub_child.attrib["lon"])
                    )
                )

        return cls(
            attributes=attributes,
            geometry=geometry,
            ref=ref,
            role=role,
            result=result
        )


class RelationNode(RelationMember):
    _type_value = "node"

    def resolve(self, resolve_missing=False):
        return self._result.get_node(self.ref, resolve_missing=resolve_missing)

    def __repr__(self):
        return "<overpy.RelationNode ref={} role={}>".format(self.ref, self.role)


class RelationWay(RelationMember):
    _type_value = "way"

    def resolve(self, resolve_missing=False):
        return self._result.get_way(self.ref, resolve_missing=resolve_missing)

    def __repr__(self):
        return "<overpy.RelationWay ref={} role={}>".format(self.ref, self.role)


class RelationWayGeometryValue(object):
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def __repr__(self):
        return "<overpy.RelationWayGeometryValue lat={} lon={}>".format(self.lat, self.lon)


class RelationRelation(RelationMember):
    _type_value = "relation"

    def resolve(self, resolve_missing=False):
        return self._result.get_relation(self.ref, resolve_missing=resolve_missing)

    def __repr__(self):
        return "<overpy.RelationRelation ref={} role={}>".format(self.ref, self.role)


class RelationArea(RelationMember):
    _type_value = "area"

    def resolve(self, resolve_missing=False):
        return self._result.get_area(self.ref, resolve_missing=resolve_missing)

    def __repr__(self):
        return "<overpy.RelationArea ref={} role={}>".format(self.ref, self.role)


class OSMSAXHandler(handler.ContentHandler):
    """
    SAX parser for Overpass XML response.
    """
    #: Tuple of opening elements to ignore
    ignore_start = ('osm', 'meta', 'note', 'bounds', 'remark')
    #: Tuple of closing elements to ignore
    ignore_end = ('osm', 'meta', 'note', 'bounds', 'remark', 'tag', 'nd', 'center')

    def __init__(self, result):
        """
        :param result: Append results to this result set.
        :type result: overpy.Result
        """
        handler.ContentHandler.__init__(self)
        self._result = result
        self._curr = {}
        #: Current relation member object
        self.cur_relation_member = None

    def startElement(self, name, attrs):
        """
        Handle opening elements.

        :param name: Name of the element
        :type name: String
        :param attrs: Attributes of the element
        :type attrs: Dict
        """
        if name in self.ignore_start:
            return
        try:
            handler = getattr(self, '_handle_start_%s' % name)
        except AttributeError:
            raise KeyError("Unknown element start '%s'" % name)
        handler(attrs)

    def endElement(self, name):
        """
        Handle closing elements

        :param name: Name of the element
        :type name: String
        """
        if name in self.ignore_end:
            return
        try:
            handler = getattr(self, '_handle_end_%s' % name)
        except AttributeError:
            raise KeyError("Unknown element end '%s'" % name)
        handler()

    def _handle_start_center(self, attrs):
        """
        Handle opening center element

        :param attrs: Attributes of the element
        :type attrs: Dict
        """
        center_lat = attrs.get("lat")
        center_lon = attrs.get("lon")
        if center_lat is None or center_lon is None:
            raise ValueError("Unable to get lat or lon of way center.")
        self._curr["center_lat"] = Decimal(center_lat)
        self._curr["center_lon"] = Decimal(center_lon)

    def _handle_start_tag(self, attrs):
        """
        Handle opening tag element

        :param attrs: Attributes of the element
        :type attrs: Dict
        """
        try:
            tag_key = attrs['k']
        except KeyError:
            raise ValueError("Tag without name/key.")
        self._curr['tags'][tag_key] = attrs.get('v')

    def _handle_start_node(self, attrs):
        """
        Handle opening node element

        :param attrs: Attributes of the element
        :type attrs: Dict
        """
        self._curr = {
            'attributes': dict(attrs),
            'lat': None,
            'lon': None,
            'node_id': None,
            'tags': {}
        }
        if attrs.get('id', None) is not None:
            self._curr['node_id'] = int(attrs['id'])
            del self._curr['attributes']['id']
        if attrs.get('lat', None) is not None:
            self._curr['lat'] = Decimal(attrs['lat'])
            del self._curr['attributes']['lat']
        if attrs.get('lon', None) is not None:
            self._curr['lon'] = Decimal(attrs['lon'])
            del self._curr['attributes']['lon']

    def _handle_end_node(self):
        """
        Handle closing node element
        """
        self._result.append(Node(result=self._result, **self._curr))
        self._curr = {}

    def _handle_start_way(self, attrs):
        """
        Handle opening way element

        :param attrs: Attributes of the element
        :type attrs: Dict
        """
        self._curr = {
            'center_lat': None,
            'center_lon': None,
            'attributes': dict(attrs),
            'node_ids': [],
            'tags': {},
            'way_id': None
        }
        if attrs.get('id', None) is not None:
            self._curr['way_id'] = int(attrs['id'])
            del self._curr['attributes']['id']

    def _handle_end_way(self):
        """
        Handle closing way element
        """
        self._result.append(Way(result=self._result, **self._curr))
        self._curr = {}

    def _handle_start_area(self, attrs):
        """
        Handle opening area element

        :param attrs: Attributes of the element
        :type attrs: Dict
        """
        self._curr = {
            'attributes': dict(attrs),
            'tags': {},
            'area_id': None
        }
        if attrs.get('id', None) is not None:
            self._curr['area_id'] = int(attrs['id'])
            del self._curr['attributes']['id']

    def _handle_end_area(self):
        """
        Handle closing area element
        """
        self._result.append(Area(result=self._result, **self._curr))
        self._curr = {}

    def _handle_start_nd(self, attrs):
        """
        Handle opening nd element

        :param attrs: Attributes of the element
        :type attrs: Dict
        """
        if isinstance(self.cur_relation_member, RelationWay):
            if self.cur_relation_member.geometry is None:
                self.cur_relation_member.geometry = []
            self.cur_relation_member.geometry.append(
                RelationWayGeometryValue(
                    lat=Decimal(attrs["lat"]),
                    lon=Decimal(attrs["lon"])
                )
            )
        else:
            try:
                node_ref = attrs['ref']
            except KeyError:
                raise ValueError("Unable to find required ref value.")
            self._curr['node_ids'].append(int(node_ref))

    def _handle_start_relation(self, attrs):
        """
        Handle opening relation element

        :param attrs: Attributes of the element
        :type attrs: Dict
        """
        self._curr = {
            'attributes': dict(attrs),
            'members': [],
            'rel_id': None,
            'tags': {}
        }
        if attrs.get('id', None) is not None:
            self._curr['rel_id'] = int(attrs['id'])
            del self._curr['attributes']['id']

    def _handle_end_relation(self):
        """
        Handle closing relation element
        """
        self._result.append(Relation(result=self._result, **self._curr))
        self._curr = {}

    def _handle_start_member(self, attrs):
        """
        Handle opening member element

        :param attrs: Attributes of the element
        :type attrs: Dict
        """

        params = {
            # ToDo: Parse attributes
            'attributes': {},
            'ref': None,
            'result': self._result,
            'role': None
        }
        if attrs.get('ref', None):
            params['ref'] = int(attrs['ref'])
        if attrs.get('role', None):
            params['role'] = attrs['role']

        cls_map = {
            "area": RelationArea,
            "node": RelationNode,
            "relation": RelationRelation,
            "way": RelationWay
        }
        cls = cls_map.get(attrs["type"])
        if cls is None:
            raise ValueError("Undefined type for member: '%s'" % attrs['type'])

        self.cur_relation_member = cls(**params)
        self._curr['members'].append(self.cur_relation_member)

    def _handle_end_member(self):
        self.cur_relation_member = None
