head_pred(row, 3).
type(row, (panel, panel, panel)).
direction(row, (in, in, in)).

body_pred(n_shapes, 2).
type(n_shapes, (panel, int)).
direction(n_shapes, (in, out)).

body_pred(shape_prop, 3).
type(shape_prop, (shape, property, val)).
direction(shape_prop, (in, out, out)).

body_pred(has_shape, 2).
type(has_shape, (panel, shape)).
direction(has_shape, (in, out)).

body_pred(select_, 3).
type(select_, (val, list, list)).
direction(select_, (out, in, out)).

body_pred(same_in_prop, 3).
type(same_in_prop, (shape, shape, property)).
direction(same_in_prop, (in, in, in)).

body_pred(identical, 2).
type(identical, (shape, shape)).
direction(identical, (in, in)).