-module(neuron).
-export([neuron/5, neuron/0]).

random() ->
    0.
% checks whether an element is in a list
listFind(Element, List) ->
    lists:member(Element, List).

 % added this line because of vague notion of erlang proper form
thresholding(Posteriors, Weights, NewNet) ->
    Threshold = 10,
    if
        NewNet > Threshold ->
            feed(Posteriors, Weights, NewNet),
            0;
        true ->
            NewNet
    end.



% global variable holder
% commmented out because of proper erlang syntax
% threshold() ->
%     10.


% the neuron process handler
% neuron([], [], [], 0, -1)
neuron() ->
    neuron([], [], [], 0, -1).
neuron(Proximals, Posteriors, Weights, Net, Marker) ->
    receive
        {feed, Num} ->
            neuron(Proximals, Posteriors, Weights, thresholding(Posteriors, Weights, Num + Net), Marker);
            % replaced code block because of erlang standards
            % NewNet = Num + Net,
            % if
            %     Num + Net > threshold() ->
            %         feed(Posterior, Weights, Num+Net),
            %         neuron(Proximals, Posteriors, Weights, 0);
            %     true ->
            %         neuron(Proximals, Posteriors, Weights, Net+Num)
            % end;

        {proximal, Neuron} ->
            neuron([Neuron | Proximals], Posteriors, Weights, Net, Marker);
        % makes a posterior connection with the neuron
        % self() -> Neuron
        {posterior, Neuron} ->
            % run a check
            case listFind(Neuron, Posteriors) of
                false ->
                    neuron(Proximals, [Neuron | Posteriors], [random() | Weights], Net, Marker);
                true ->
                    neuron(Proximals, Posteriors, Weights, Net, Marker)
            end;
        {dfs, NewMarker, Parent} ->
            if
                NewMarker > Marker ->
                    Parent ! {posterior, self()},
                    dfs(Proximals, NewMarker, self()),
                    neuron(Proximals, Posteriors, Weights, Net, NewMarker)
            end

    end.
% depth first search along the proximals
dfs([], _, _) ->
    ok;
dfs(Proximals, NewMarker, Self) ->
    [Head|Rest] = Proximals,
    Head ! {dfs, NewMarker, Self},
    dfs(Rest, NewMarker, Self).

% handles feed forwarding for a list of neurons and their corresponding weights
feed([], [], _) ->
    ok;
feed(Neurons, Weights, Net) ->
    io:format('done'),
    [NextNeur | RestNeurs]  = Neurons,
    [NextWeight | RestWeight] = Weights,
    NextNeur ! {feed, NextWeight * Net},
    feed(RestNeurs, RestWeight, Net).
