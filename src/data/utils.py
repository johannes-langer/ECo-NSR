import xmltodict


def get_xml(path: str):
    """
    Reads `path` to `dict`
    """

    assert path[-4:] == ".xml"

    with open(path, "r") as file:
        out = xmltodict.parse(file.read())

    return out


def convert_stringlist(s: str) -> list[float]:
    """
    Takes stringlist and returns list of floats. Makes no assumptions about whitespaces before or after the list.
    """
    return [float(i) for i in s[s.find("[") + 1 : s.find("]")].split(",")]
