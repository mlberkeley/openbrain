-module(brain).
-export([brain/1,brain/3, construct/3, start/0, stop/0, feed/2, pseudo_server/0]).

-record(matrix, {matrix=[],
                r=0,
                c=0}).
-record(brainParams, {  ins=[], outs=[], nodesz=0, inputsz=0, outputsz=0
                    }).


%% @doc The brain server that takes in inputs
brain(NumNodes, NumInputs, NumOutputs) ->
    brain(#brainParams{nodesz=NumNodes, inputsz=NumInputs, outputsz=NumOutputs}).
brain(Params) ->
    % io:format('~w~n', ['lit']),
    receive
        % TODO consider whether this would be better in constructor paradigm
        newbrain ->
            NewParams = construct(Params),
            brain(NewParams);
        {feed, Inputs} ->
            feed(Inputs, Params#brainParams.ins),
            brain(Params);
        % returns a list of the output neuron values
        getoutput ->
            % [Head | Tail] = Params#brainParams.outs,
            % io:format("slit~w~n", [neuron:net(Head)]),
            print_output(Params#brainParams.outs),
            % [neuron:net(X  ! getNeuron) || X <- Params#brainParams.outs],
            brain(Params);
        stop ->
            unregister(brain);

        regOne ->
            [In | _] = Params#brainParams.ins,
            In ! register,
            brain(Params)
    end.

% @doc Passes input valus into the network
feed([],_) ->
    ok;
feed(_,[]) ->
    ok;
feed(InputVals, InputPIDs) ->

    [PID| RestPID] = InputPIDs,
    [Val | RestVal] = InputVals,
    % io:format("feed ~w -> ~w~n",[Val, PID]),
    PID ! {feed, Val},
    feed(RestVal, RestPID).




%% @doc Randomly assigns some numbers as input and output neurons based on the
%% size of the input and output vectors
%% @params MatrixRcr
assign_neurons(NumInputs, NumOutputs, MatrixRcrd) ->
    L = lists:seq(1, trunc(math:pow(ncols(MatrixRcrd),2))), % TODO this should be the number of elements in hte graph
    Shuffle = [X||{_,X} <- lists:sort([ {random:uniform(), N} || N <- L])],
    Ins = get_input_pids(lists:sublist(Shuffle, NumInputs),MatrixRcrd),
    Outs = get_output_pids(lists:sublist(Shuffle, NumInputs+1, NumOutputs),MatrixRcrd),

    % returns
    {Ins, Outs}.

%% @doc Returns the process ids for the input neurons indexed by Positions.
get_input_pids(Positions, MatrixRcrd) ->
    % Matrix = matrix(MatrixRcrd),
    get_input_pids(Positions, MatrixRcrd, []).
get_input_pids([], _, PiDs) ->
    PiDs;
get_input_pids(Positions,  Matrix, PiDs) ->
    [CurPosition| RestPositions] = Positions,
    NewPid = access(Matrix, CurPosition),
    NewPid ! {set_type, input},
    get_input_pids(RestPositions, Matrix, [NewPid | PiDs]).

%% @doc Returns the process ids for the output neurons indexed by Positions.
get_output_pids(Positions, MatrixRcrd) ->
    % Matrix = matrix(MatrixRcrd),
    get_output_pids(Positions, MatrixRcrd, 0, []).
get_output_pids([], _, _, PiDs)->
    PiDs;
get_output_pids(Positions, Matrix, Count, PiDs) ->
    [CurPosition| RestPositions] = Positions,
    NewPid = access(Matrix, CurPosition),
    NewPid ! {set_type, output, Count},
    get_output_pids(RestPositions, Matrix, Count + 1, [NewPid|PiDs]).
% getPids([], OutputL, _) ->
%     OutputL;
% getPids(Positions, OutputL, Matrix) ->
%     [CurPosition| RestPositions] = Positions,
%     {X,Y} = list_to_matrix(CurPosition, nrows(Matrix), ncols(Matrix)),
%     getPids(RestPositions, [access(Matrix, X, Y)|OutputL], Matrix).

% returns the Pids of all and marks some as output
% getPids(output, [], OutputL, _, _) ->
%     OutputL;
% getPids(output,InputL, OutputL, MatrixRcrd, Count) ->
%     [FirstI| RestI] = InputL,
%     {X,Y} = list_to_matrix(FirstI, nrows(MatrixRcrd), ncols(MatrixRcrd)),
%     NewPid = access(MatrixRcrd, X, Y),
%     NewPid ! {output,Count},
%     getPids(output, RestI, [NewPid| OutputL], MatrixRcrd, Count + 1).

%% Creates a graph that matches the nodal structure
construct(BrainParams) ->
    #brainParams{inputsz=InputC, outputsz=OutputC, nodesz=NodeC} = BrainParams,
    construct(NodeC, InputC, OutputC).


%% @doc Creates the network with NumNodes nodes, NumInputs inputs and NumOutputs
%% outputs. NumNodes > NumInput + NumOutputs.
construct(NumNodes, NumInputs, NumOutputs) ->
    % #brainParams{inputsz=NumInputs, outputsz=NumOutputs, nodesz=NumNodes} = BrainParams,
    Side = ceiling(math:sqrt(NumNodes)),
    Params = #brainParams{inputsz=NumInputs, outputsz=NumOutputs, nodesz=Side*Side},

    ProxMat = gen_matrix(Side,Side),
    % Matrix = matrix(ProxMat),
    % setup proximity graph

    proximity_rows(ProxMat),

    % pull out the input and output lists
    {InNeurons, OutNeurons} = assign_neurons(NumInputs, NumOutputs, ProxMat),
    % Construct the Connectivity graph
    create_connections(InNeurons, 0),
    % io:format("post-create_connections~n",[]),

    % copies the
    Params#brainParams{ins=InNeurons, outs=OutNeurons}.

%% @doc Starts the depth first search that creates the feedforward graph
create_connections([], _) ->
    ok;
create_connections(InputNL, Counter) ->
    [Head | Rest] = InputNL,
    io:format("Starting ~w Input's DFS~n",[Counter]),
    Head ! {startdfs, Counter},
    create_connections(Rest, Counter+1).

%% @doc Sets up the proximity graph for all of the elements in the matrix.
proximity_rows(MatrixRcrd) ->
    proximity_rows(matrix(MatrixRcrd), ncols(MatrixRcrd), nrows(MatrixRcrd), MatrixRcrd).
proximity_rows([],_,_, _) ->
    ok;
proximity_rows(Matrix, RowIdx, Length, MatrixRcrd) ->
    [Row| Rest] = Matrix,
    proximity_row_elms(Row, RowIdx, Length, MatrixRcrd),
    proximity_rows(Rest, RowIdx-1, Length, MatrixRcrd).

%% @doc Sets up the proximity graph of all of the rows.
%% Helper to proximity_rows
%% @see proximity_rows
proximity_row_elms([], _, _, _) ->
    ok;
proximity_row_elms(List, Row, Col, Matrix) ->
    [First | Rest] = List,
    set_proximity(First, get_adjacents(Matrix, Row, Col)),
    proximity_row_elms(Rest, Row, Col-1, Matrix).

%% --------------------------------
%% TODO MOVE THIS TO NEURON
%% @doc Sets up the links from the node to its immediately proximal neurons
%% Helper to proximity_row_elms.
%% @see proximity_row_elms
set_proximity(_, []) ->
    ok;
set_proximity(Node, Adjacents) ->
    [First | Rest] = Adjacents,
    Node ! {proximal, First},
    % io:format("~w -> ~w~n", [Node, First]),
    set_proximity(Node, Rest).

% returns the 8 adjacent matrices to the node at this position
%% TODO make this less janky
get_adjacents(MatrixRcrd, R, C) ->
    [access(MatrixRcrd, R+1, C),
    access(MatrixRcrd, R+1, C+1),
    access(MatrixRcrd, R, C+1),
    access(MatrixRcrd, R-1, C+1),
    access(MatrixRcrd, R-1, C),
    access(MatrixRcrd, R-1, C-1),
    access(MatrixRcrd, R, C-1),
    access(MatrixRcrd, R+1, C-1)].
%% -------------------------
%% -------------------------
%% Test utils
print_output([]) ->
    ok;
print_output(L) ->
    [Head | Rest] = L,
    Head ! identity,
    % io:format("this~w~n",[(Head ! getNeuron)]),
    print_output(Rest).
%% -------------------------
%% --- move all of this to a math util
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

% modulus operator
mod(X,Y) ->
    R = X rem Y,
    if R < 0 ->
        R + Y;
    true ->
        R
    end.

%% --- end of move

%% -----------------------
%% MOVE THIS TO A MATRIX MODULE
% Matrix data structure function
% generates the matrix
gen_matrix(Rows,Columns) ->
    #matrix{matrix=[[spawn(neuron, neuron, []) || _ <- lists:seq(1,Columns)] || _ <- lists:seq(1,Rows)], r=Rows, c=Columns}.

% gets the matrix component of the matrix data structure
matrix(MatrixRcrd) ->
    % io:format("matrix()~w~n",[MatrixRcrd]),
    MatrixRcrd#matrix.matrix.

% gets the number of columns of a matrix record
ncols(MatrixRcrd) ->
    % io:format("matrix()~w~n",[MatrixRcrd]),
    MatrixRcrd#matrix.c.

%gets the number of columns of a matrix record
nrows(MatrixRcrd) ->
    MatrixRcrd#matrix.r.

%% Accesses matrix elements using 1d indexing
access(MatrixRcrd, I) ->
    {X, Y} = list_to_matrix(I, nrows(MatrixRcrd),ncols(MatrixRcrd)),
    access(MatrixRcrd, X, Y).

% grabs the element at row R and column C in Matrix.
% 1 indexed !!! TODO make this 0 indexed
access(MatrixRcrd, X, Y) ->
    Matrix = matrix(MatrixRcrd),
    lists:nth(
        mod(Y, ncols(MatrixRcrd))+1,
        lists:nth(mod(X, nrows(MatrixRcrd))+1, Matrix)).

%% @doc Converts a 1d position to the coordinates in a 2d_matrix with dims MxN
list_to_matrix(Pos, M, N) ->
    X = floor(Pos/M) + 1,
    Y = Pos rem N + 1,
    {X, Y}.
%% ---- end  of the module

pseudo_server() ->
    receive
        Other ->
            io:format("received ~w~n",[Other])
    end.
% c(brain) and c(neuron) first
start() ->
    % compile:file(neuron),
    % brain ! stop,
    game:start(),
    register(brain, spawn(brain, brain, [10,2,2])),
    brain ! newbrain,
    register(pseudojava, spawn(brain, pseudo_server, [])),
    inputListener ! {connect, pseudojava}.
    % brain
    % brain ! regOne,
    % brain ! getoutput.
% stops the current running brain.
stop() ->
    game:stop(),
    unregister(brain).

    % brain ! {feed, [1,1]}.

    % brain ! {feed, [1,1]}.





% creates the proximal graphs
% construct_graph(Rows, Columns) ->
    % asdkfjlkajsd;flkjas;dlkfj;aksdjf;akjsd;fkljasd;fklja;skdjf;aksjdf;kjasd;fkja;sdkjf;aksjdf;kasjd

% Creates the brain cluster
% create_brain(Nodes) ->
%     TempColumn = Nodes / 3,
%     Column = max(trunc(Column), round(Column)),
%     construct_matrix(3, Column).
