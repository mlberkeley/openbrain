-module(neuron).
-export([neuron/1, neuron/0, net/1]).

-record(neuron, {proximals=[],
                posteriors=[],
                weights=[],
                net=0,
                marker=-1,
                markerProx=[],
                output=false}). % the proximals that have not been checked by posteriors

%  Neuron **SHOULD BE**  a neuron, but is not
net(Neuron) ->
    Neuron#neuron.net.
% shuffle a list randomly
shuffle(L) ->
    [X||{_,X} <- lists:sort([ {random:uniform(), N} || N <- L])].

% checks whether an element is in a list
listFind(Element, List) ->
    lists:member(Element, List).

 % added this line because of vague notion of erlang proper form
thresholding(Posteriors, Weights, NewNet) ->
    Threshold = 0.5,
    if
        NewNet > Threshold ->
            feed(Posteriors, Weights, NewNet),
            0;
        true ->
            NewNet
    end.

% the neuron process handler
% neuron([], [], [], 0, -1)
addProximal(Neuron, Proximal) ->
    Proximals = Neuron#neuron.proximals,
    Neuron#neuron{proximals=[Proximal|Proximals]}.
addPosterior(Neuron, Posterior) ->
    Posteriors = Neuron#neuron.posteriors,
    case listFind(Posterior, Posteriors) of
        false ->
            Neuron#neuron{posteriors=[Posterior|Posteriors], weights=[min(random:uniform(), 0.99) |  Neuron#neuron.weights]};
        true ->
            Neuron
    end.

neuron() ->
    neuron(#neuron{}).
% neuron(Proximals, Posteriors, Weights, Net, Marker) ->
neuron(Neuron) ->
    receive
        % feed forward structure
        printNet ->
            io:format("Net: ~w~n", [net(Neuron)]),
            neuron(Neuron);
        {feed, Num} ->
            Posteriors = Neuron#neuron.posteriors,
            Weights = Neuron#neuron.weights,
            Net = Neuron#neuron.net,
            %io:format("{~w, Net:~w}~n",[self(),Net+Num]),
            neuron(Neuron#neuron{net=thresholding(Posteriors, Weights, Num+Net)});
        % adds a proximal node to this neuron's graph
        {proximal, Prox} ->
            neuron(addProximal(Neuron, Prox));

        % makes a posterior connection with the neuron. If it already exists, then
        % nothing happens
        {posterior, Posterior} ->
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
            #neuron{marker=Marker, output=IsOutput} = Neuron,
            if
                % stop running dfs if we hit output node
                IsOutput == true ->
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
        % flags this node as output
        output ->
            neuron(Neuron#neuron{output=true})

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
