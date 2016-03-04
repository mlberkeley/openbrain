-module(neuron).
-export([neuron/1, neuron/0, net/1]).

-record(neuron, {proximals=[],
                posteriors=[],
                weights=[],
                net=0,
                marker=-1,
                markerProx=[],
                type=normal,
                key=-1}).
                % output={false, -1}}). % the proximals that have not been checked by posteriors

net(NeuronRcrd) ->
    NeuronRcrd#neuron.net.
% shuffle a list randomly
shuffle(L) ->
    [X||{_,X} <- lists:sort([ {random:uniform(), N} || N <- L])].

% checks whether an element is in a list
listFind(Element, List) ->
    lists:member(Element, List).

%% @doc Handles the thresholding action of net.
thresholding(Posteriors, Weights, NewNet) ->
    Threshold = 1,
    if
        NewNet > Threshold ->
            feed(Posteriors, Weights, NewNet),
            0;
        true ->
            NewNet
    end.
%% @TODO this could prboably be incorporated into feed.
% threshold([], [], NewNet) - this is probably the solution
thresholding(NewNet, Key) ->
    Threshold = 0.5,
	io:format("uber eveywhere ~w~n", ["shorty bad as hell"]),
    if
        NewNet > Threshold ->
            pressKey(Key),
            0;
        true ->
            NewNet
    end.


% the neuron process handler
% neuron([], [], [], 0, -1)
addProximal(Neuron, Proximal) ->
    Proximals = Neuron#neuron.proximals,
    Neuron#neuron{proximals=[Proximal|Proximals]}.
addPosterior(Neuron, NewPost) ->
    Posteriors = Neuron#neuron.posteriors,
    case listFind(NewPost, Posteriors) of
        false ->
            Neuron#neuron{posteriors=[NewPost|Posteriors], weights=[min(random:uniform(), 0.99) |  Neuron#neuron.weights]};
        true ->
            Neuron
    end.

neuron() ->
    neuron(#neuron{}).
% neuron(Proximals, Posteriors, Weights, Net, Marker) ->
neuron(Neuron) ->
    receive
        % feed forward structure
        regSelf ->
            io:format('tryna register ~n',[]),
            global:re_register_name(neuron, self()),
            neuron(Neuron);
        identity ->
            io:format("Neuron: ~w~n", [Neuron]),
            neuron(Neuron);
        printNet ->
            io:format("Net: ~w~n", [net(Neuron)]),
            neuron(Neuron);
        {feed, Num} ->
			io:format("yuh ~w~n", [self()]),
            #neuron{posteriors=Posteriors, weights=Weights, net=Net, type=Type, key=Key} = Neuron,
            % Posteriors = Neuron#neuron.posteriors,
            % Weights = Neuron#neuron.weights,
            % Net = Neuron#neuron.net,
            % io:format("{~w, Fed:~w}~n",[self(),Num]),
            % {IsOutput, Key} = Outputs,
            Is_Output = (Type == output),
            if
                Is_Output == false ->
                    neuron(Neuron#neuron{net=thresholding(Posteriors, Weights, Num+Net)});
                Is_Output == true ->
                    neuron(Neuron#neuron{net=thresholding(Num+Net, Key)})
            end;
        % adds a proximal node to this neuron's graph
        {proximal, Prox} ->
            neuron(addProximal(Neuron, Prox));
        % makes a posterior connection with the neuron. If it already exists, then
        % nothing happens
        {posterior, Posterior} ->
            % io:format('Add proximal ~w->~w~n',[self(), Posterior]),
            neuron(addPosterior(Neuron, Posterior));
        % if the dfs of as child node fails, then
        {faileddfs, Touched, Marker} ->
            MarkerProximals= Neuron#neuron.markerProx,
            NewNeuron = Neuron#neuron{markerProx=lists:delete(Touched, MarkerProximals)},
            dfsNext(NewNeuron, Marker, self()),
            neuron(NewNeuron);
        {startdfs, NewMarker} ->
            dfsNext(resetDFS(Neuron), NewMarker, self()),
            neuron(Neuron);
        % the dfs receiver
        {dfs, NewMarker, ParentPID} ->
            #neuron{marker=Marker, type=Type} = Neuron,
            IsOutput = (Type == output),
            if
                % stop running dfs if we hit output node
                IsOutput /= false ->
                    ParentPID ! {posterior, self()},
                    neuron(Neuron);
                % has not been marked by this node yet
                NewMarker > Marker ->
                    % notify the parent that this is a good connection
                    ParentPID ! {posterior, self()},
                    % reset the MarkProximals s othat they will randomly choose
                    dfsNext(resetDFS(Neuron), NewMarker, self()),
                    neuron(Neuron);
                %  has been marked
                true ->
                    ParentPID ! {faileddfs, self(), NewMarker},
                    neuron(Neuron)
            end;
        % flags this node as Type
        {set_type, Type} ->
            neuron(Neuron#neuron{type=Type});
        % special handle for output
        {set_type, output, Key} ->
            neuron(Neuron#neuron{type=output,key=Key});
		stop ->
			ok



    end.
%% ----------------------------------------
%% TODO MOVE THIS TO ANOTHER MODULE
%% @doc Sends the pressKey command to the mc server.
pressKey(KeyNum) ->
	io:format("pressed ~w~n", [KeyNum]),
    input_listener ! inputKey(KeyNum).
%% @doc Maps key index to key press
inputKey(KeyNum) ->
	
    case KeyNum of
        
        0 -> w;
        1 -> a;
        2 -> s;
        3 -> d;
        4 -> space;
		_ -> w % remove this
    end.

% reset the MarkProximals s othat they will randomly choose
resetDFS(Neuron) ->
    MarkerProx = shuffle(Neuron#neuron.proximals),
    Neuron#neuron{markerProx=MarkerProx}.
dfsNext(#neuron{markerProx=[]}, _, _) ->
    ok;
dfsNext(#neuron{markerProx=Proximals}, Marker, ParentPID) ->
    [Head | _] = Proximals,
    Head ! {dfs, Marker, ParentPID}.

% handles feed forwarding for a list of neurons and their corresponding weights
feed([], [], _) ->
    ok;
feed(Neurons, Weights, Net) ->
    [NextNeur | RestNeurs]  = Neurons,
    [NextWeight | RestWeight] = Weights,
    NextNeur ! {feed, NextWeight * Net},
    feed(RestNeurs, RestWeight, Net).
