-module(brain).
-export([brain/0, brain/1,brain/3, construct/3, main/0]).

-record(matrix, {matrix=[],
                r=0,
                c=0}).
-record(brainParams, {  ins=[],
                        outs=[],
                        nodesz=0,
                        inputsz=0,
                        outputsz=0
                    }).


% the main server
brain() ->
    brain(#brainParams{}).
brain(Nodesz, Inputsz, Outputsz) ->
    brain(#brainParams{nodesz=Nodesz, inputsz=Inputsz, outputsz=Outputsz}).
brain(Params) ->
    % io:format('~w~n', ['lit']),
    receive
        newbrain ->
            NewParams = construct(Params),
            brain(NewParams);
        {feed, Inputs} ->
            feed(Params#brainParams.ins, Inputs),
            brain(Params);
        % returns a list of the output neuron values
        getoutput ->
            % [Head | Tail] = Params#brainParams.outs,
            % io:format("slit~w~n", [neuron:net(Head)]),
            printNets(Params#brainParams.outs),
            % [neuron:net(X  ! getNeuron) || X <- Params#brainParams.outs],
            brain(Params)
    end.
printNets([]) ->
    ok;
printNets(L) ->
    [Head | Rest] = L,
    Head ! printNet,
    % io:format("this~w~n",[(Head ! getNeuron)]),
    printNets(Rest).
% feeds the network input values
feed([],_) ->
    ok;
feed(_,[]) ->
    ok;
feed(InputPIDs, InputVals) ->
    [PID| RestPID] = InputPIDs,
    [Val | RestVal] = InputVals,
    PID ! {feed, Val},
    feed(RestPID, RestVal).


% function which rounds a number to the nearest int <= x
floor(X) when X < 0 ->
    T = trunc(X),
    case X - T == 0 of
        true -> T;
        false -> T - 1
    end;
floor(X) ->
    trunc(X).

% function which rounds a number to the nearest int >= x
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

% returns random values
get_ins_outs(InputC, OutputC, MatrixRec) ->
    L = lists:seq(1, trunc(math:pow(ncols(MatrixRec),2))),
    Shuffle = [X||{_,X} <- lists:sort([ {random:uniform(), N} || N <- L])],
    Ins = getPids(lists:sublist(Shuffle, InputC),[], MatrixRec),
    Outs = getPids(output,lists:sublist(Shuffle, InputC+1, OutputC),[], MatrixRec),
    {Ins, Outs}.

% converts the position value from a 1d vector to a 2d of dimensions MxN
dto2d(Pos, M, N) ->
    X = floor(Pos/M),
    Y = Pos rem N,
    {X,Y}.

% returns the Pids of all
getPids([], OutputL, _) ->
    OutputL;
getPids(InputL, OutputL, MatrixRec) ->
    [FirstI| RestI] = InputL,
    {X,Y} = dto2d(FirstI, nrows(MatrixRec), ncols(MatrixRec)),
    getPids(RestI, [access(MatrixRec, X, Y)|OutputL], MatrixRec).

% returns the Pids of all and marks some as output
getPids(output, [], OutputL, _) ->
    OutputL;
getPids(output,InputL, OutputL, MatrixRec) ->
    [FirstI| RestI] = InputL,
    {X,Y} = dto2d(FirstI, nrows(MatrixRec), ncols(MatrixRec)),
    NewPid = access(MatrixRec, X, Y),
    NewPid ! output,
    getPids(output, RestI, [NewPid| OutputL], MatrixRec).
% % Creates a graph that matches the nodal structure
construct(BrainParams) ->
    #brainParams{inputsz=InputC, outputsz=OutputC, nodesz=NodeC} = BrainParams,
    construct(NodeC, InputC, OutputC).
construct(NodeC, InputC, OutputC) ->
    % #brainParams{inputsz=InputC, outputsz=OutputC, nodesz=NodeC} = BrainParams,
    Side = ceiling(math:sqrt(NodeC)),
    BrainParams = #brainParams{inputsz=InputC, outputsz=OutputC, nodesz=Side*Side},

    ProxMat = gen_matrix(Side,Side),
    % Matrix = matrix(ProxMat),
    % setup proximity graph
    io:format("pre-proximity~n",[]),
    proximity_rows(matrix(ProxMat), Side, Side, ProxMat),
    io:format("post-proximity~n",[]),
    % pull out the input and output lists
    {Ins, Outs} = get_ins_outs(InputC, OutputC, ProxMat),
    % Construct the Connectivity graph
    connectivity(Ins, 0),
    % io:format("post-connectivity~n",[]),
    BrainParams#brainParams{ins=Ins, outs=Outs}.

% handles the dfs connectivity
connectivity([], _) ->
    ok;
connectivity(InputNL, Counter) ->
    [Head | Rest] = InputNL,
    Head ! {startdfs, Counter},
    connectivity(Rest, Counter+1).

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
% c(brain) and c(neuron) first
main() ->
    % c(neuron),
    % unregister(brain),
    register(brain, spawn(brain, brain, [200000,8,4])),
    brain ! newbrain,
    brain ! getoutput.





% creates the proximal graphs
% construct_graph(Rows, Columns) ->
    % asdkfjlkajsd;flkjas;dlkfj;aksdjf;akjsd;fkljasd;fklja;skdjf;aksjdf;kjasd;fkja;sdkjf;aksjdf;kasjd

% Creates the brain cluster
% create_brain(Nodes) ->
%     TempColumn = Nodes / 3,
%     Column = max(trunc(Column), round(Column)),
%     construct_matrix(3, Column).
