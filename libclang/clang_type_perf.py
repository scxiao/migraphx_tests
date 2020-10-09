#!/usr/bin/env python
"""call with <filename> <typename>"""

import sys
import clang.cindex as clang

def find_typerefs(node, typename):
    """ Find all references to the type named 'typename'
    """
    print("node_name = {}".format(node.kind))
#    if node.kind.is_reference():
#    if node.kind == clang.CursorKind.VAR_DECL:
        #ref_node = clang.cindex.Cursor_ref(node)
    ref_node = node
#if ref_node.spelling == typename:
    print('Found {} [line={}, col={}]'.format(
          node.spelling, node.location.line, node.location.column))
    print()
    # Recurse for children of this node
    for c in node.get_children():
        find_typerefs(c, typename)

index = clang.Index.create()
tu = index.parse(sys.argv[1])
print('Translation unit: {}'.format(tu.spelling))
find_typerefs(tu.cursor, sys.argv[2])

