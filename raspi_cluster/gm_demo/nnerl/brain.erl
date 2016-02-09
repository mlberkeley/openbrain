-module(brain).
-export([construct/3, access/3, get_adjacents/3, matrix/1, ncols/1, nrows/1]).

-record(matrix, {matrix=[], r=0, c=0}).



% floor(X) when X < 0 ->
%     T = trunc(X),
%     case X - T == 0 of
%         true -> T;
%         false -> T - 1
%     end;
% floor(X) ->
%     trunc(X).


ceiling(X) when X < 0 ->
    trunc(X);
ceiling(X) ->
    T = trunc(X),
    case X - T == 0 of
        true -> T;
        false -> T + 1
    end.

% modulus
mod(X,Y) ->
    R = X rem Y,
    if R < 0 ->
        R + Y;
    true ->
        R
    end.

% Matrix data structure function
% generates the matrix
gen_matrix(Rows,Columns) ->
    #matrix{matrix=[[spawn(neuron, neuron, []) || _ <- lists:seq(1,Columns)] || _ <- lists:seq(1,Rows)], r=Rows, c=Columns}.

% gets the matrix component of the matrix data structure
matrix(MatrixRec) ->
    % io:format("matrix()~w~n",[MatrixRec]),
    MatrixRec#matrix.matrix.
% gets the number of columns of a matrix record
ncols(MatrixRec) ->
    % io:format("matrix()~w~n",[MatrixRec]),
    MatrixRec#matrix.c.
%gets the number of columns of a matrix record
nrows(MatrixRec) ->
    MatrixRec#matrix.r.

% grabs the element at row R and column C in Matrix.
% 1 indexed !!!
access(MatrixRec, R, C) ->
    Matrix = matrix(MatrixRec),
    lists:nth(
        mod(C, ncols(MatrixRec))+1,
        lists:nth(mod(R, nrows(MatrixRec))+1, Matrix)).

% Creates a graph that matches the nodal structure
construct(NodeC, InputC, OutputC) ->
    Side = ceiling(math:sqrt(NodeC)),
    ProxMat = gen_matrix(Side,Side),
    % Matrix = matrix(ProxMat),
    % setup adjacency
    proximity_rows(matrix(ProxMat), Side, Side, ProxMat)
    % pull out the input and output matrices
    .

% column-wise adjacency adjustment
proximity_rows([],_,_, _) ->
    ok;
proximity_rows(MatrixRest, Rows, Length, Matrix) ->
    [First | Rest] = MatrixRest,
    proximity_row_elms(First, Rows, Length, Matrix),
    proximity_rows(Rest, Rows-1, Length, Matrix).

% creates the adjacency of an element of matrices
% helper to proximity_rows
proximity_row_elms([], _, _, _) ->
    ok;
proximity_row_elms(List, Row, Col, Matrix) ->
    [First | Rest] = List,
    set_proximity(First, get_adjacents(Matrix, Row, Col)),
    proximity_row_elms(Rest, Row, Col-1, Matrix).

% sets the proximity of the matrix
% helper to proximity_row_elms
set_proximity(_, []) ->
    ok;
set_proximity(Node, Adjacents) ->
    [First | Rest] = Adjacents,
    Node ! {proximal, First},
    % io:format("~w -> ~w~n", [Node, First]),
    set_proximity(Node, Rest).

% returns the 8 adjacent matrices to the node at this position
get_adjacents(MatrixRec, R, C) ->
    [access(MatrixRec, R+1, C),
    access(MatrixRec, R+1, C+1),
    access(MatrixRec, R, C+1),
    access(MatrixRec, R-1, C+1),
    access(MatrixRec, R-1, C),
    access(MatrixRec, R-1, C-1),
    access(MatrixRec, R, C-1),
    access(MatrixRec, R+1, C-1)].







% creates the proximal graphs
% construct_graph(Rows, Columns) ->
    % asdkfjlkajsd;flkjas;dlkfj;aksdjf;akjsd;fkljasd;fklja;skdjf;aksjdf;kjasd;fkja;sdkjf;aksjdf;kasjd

% Creates the brain cluster
% create_brain(Nodes) ->
%     TempColumn = Nodes / 3,
%     Column = max(trunc(Column), round(Column)),
%     construct_matrix(3, Column).
