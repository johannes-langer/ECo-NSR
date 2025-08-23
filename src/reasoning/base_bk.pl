% enable discontiguous for facts
:- discontiguous n_shapes/3.
:- discontiguous shape_prop/3.
:- discontiguous has_shape/2.

select(A, B, C).

same_in_prop(S, T, P) :- shape_prop(S, P, V), shape_prop(T, P, V).

same_in_all(S, T, [P]) :- same_in_prop(S, T, P).
same_in_all(S, T, [P|R]) :- same_in_prop(S, T, P), same_in_all(S, T, R).

indentical(S, T) :- findall(P, shape_prop(S, P, _), L), same_in_all(S, T, L).
