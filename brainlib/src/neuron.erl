-module(neuron).
-export([neuron/1, neuron/0, net/1]).

-record(neuron, {proximals=[],  % the set of neurons that are in proximity, but are not connected
                posteriors=[],  % the set of neurons that are at the posterior of this neuron -> fed into by the neuron
                weights=[],     % the set of weights corresponding to the posterior
                net=0,          % the net accumulated on this neuron
                marker=-1,      % marker used for dfs
%%                markerProx=[],  % the proximity marker
                type=normal,    % the type of neuron this is.
                key=-1}).       % the key that is presssed by this neuron (when it is an output)
%% @doc Gets hte net of a neuron record.
net(NeuronRcrd) ->
    NeuronRcrd#neuron.net.
%% @doc shuffle a list
shuffle(L) ->
    [X||{_,X} <- lists:sort([ {random:uniform(), N} || N <- L])].

%% @TODO Make this pattern matching instead
%% @doc Handles the thresholding action of net.
thresholding(Posteriors, Weights, NewNet) ->
    Threshold = 0.4,
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
    Threshold = 0.2,
    if
        NewNet > Threshold ->
            pressKey(Key),
            0;
        true ->
            NewNet
    end.

%% @doc removes the proximal from the neuron.
remove_proximal(Neuron, ProximalToEnd) ->
  try Neuron#neuron.proximals of
    _ ->
      Proximals = Neuron#neuron.proximals,
      Neuron#neuron{proximals=lists:delete(ProximalToEnd, Proximals)}

  catch
    _:Error -> io:format("Neuron: ~w failed by error ~w", [Neuron, Error])
  end.

%% @doc Returns an updated record that contains the new proximal as a reference.
add_proximal(Neuron, Proximal) ->
    Proximals = Neuron#neuron.proximals,
    Neuron#neuron{proximals=[Proximal|Proximals]}.
%% @doc Returns a random weight value.
new_weight() ->
  min(random:uniform(), 0.99).
%% @doc Returns an updated record that adds an element to the posterior list
%% and removes it from the proximals of this record
add_posterior(Neuron, NewPost) ->
  Posteriors = Neuron#neuron.posteriors,
  NewNeuron = remove_proximal(Neuron, NewPost),
  NewNeuron#neuron{posteriors=[NewPost|Posteriors], weights=[new_weight()|Neuron#neuron.weights]}.

%% @doc Returns a neuron that sets one of it's proximal neurons as a posterior
dfs(Neuron, Pid) ->
  #neuron{proximals=Proximals} = shuffle_proximals(Neuron),
  case Proximals of
    [] ->
      Neuron;
    _ ->
      [NewPosterior | RProximals] = Proximals,
      add_posterior(Neuron, NewPosterior)#neuron{proximals = RProximals},
      NewPosterior ! {dfs, Pid}
  end.


  
%% @doc returns whether this neuron is an output or not
is_output(type, Type) ->
  (Type == output).
is_output(#neuron{type=Type}) ->
  (Type == output).
%% @doc starts a neuron that has no properties.
neuron() ->
    neuron(#neuron{}).
newnet(Num, Net) ->
  try Num + Net of
    _ -> Num + Net
  catch
    _:Throw ->
      io:format("~w+~w threw ~w", [Num, Net, Throw]),
      0
  end.
neuron(Neuron) ->
  receive
    {feed, Num} ->
      #neuron{posteriors=Posteriors, weights=Weights, net=Net, type=Type, key=Key} = Neuron,

      NewNet = newnet(Num, Net),
      case is_output(type, Type) of
          false ->
              neuron(Neuron#neuron{net=thresholding(Posteriors, Weights, NewNet)});
          true ->
              neuron(Neuron#neuron{net=thresholding(NewNet, Key)})
      end;
    % adds a proximal node to this neuron's graph
    {proximal, Prox} ->
      neuron(add_proximal(Neuron, Prox));
    % makes a posterior connection with the neuron.
    {posterior, Posterior} ->
      io:format('Add proximal ~w->~w~n',[self(), Posterior]),
      neuron(add_posterior(Neuron, Posterior));
    startdfs ->
      io:format('starting dfs at ~w~n',[self()]),
      NewNeuron= dfs(Neuron, self()),
      neuron(NewNeuron);
    {dfs, Anterior} ->

      NewNeuron = remove_proximal(Neuron, Anterior),
      case is_output(Neuron) of
        false ->
          neuron(dfs(NewNeuron, self()));
        true ->
          io:format('Reached output at ~w~n', [self()]), % for DEBUG
          neuron(NewNeuron)
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
  timer:sleep(200),
  Key = inputKey(KeyNum),
  io:format("pressed ~w on ~n", [KeyNum]),
  input_listener ! Key.
%% @doc Maps key index to key press
inputKey(KeyNum) ->
    case KeyNum of
        0 -> w;
        1 -> a;
        2 -> s;
        3 -> d;
        4 -> space;
		    _ -> w
    end.

%% @doc shuffles proximals so that we choose
shuffle_proximals(Neuron) ->
    NewProximals = shuffle(Neuron#neuron.proximals),
    Neuron#neuron{proximals=NewProximals}.

% handles feed forwarding for a list of neurons and their corresponding weights
feed([], [], _) ->
    ok;
feed(Neurons, Weights, Net) ->
    [NextNeur | RestNeurs]  = Neurons,
    [NextWeight | RestWeight] = Weights,
    NextNeur ! {feed, NextWeight * Net},
    feed(RestNeurs, RestWeight, Net).
