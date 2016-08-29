%%%-------------------------------------------------------------------
%%% @author Phillip Kuznetsov
%%% @copyright (C) 2016, Machine Learning at Berkeley
%%% @doc
%%%
%%% @end
%%% Created : 10. Mar 2016 5:28 PM
%%%-------------------------------------------------------------------
-module(neuron).
-author("Phillip Kuznetsov").

%% API
-export([]).
-callback fire(Neuron :: term()) ->
  ok.

-callback update_internal_state(InternalState :: tuple(),
    TimeStep :: non_neg_integer(), ReceivedInputs :: list(term())) ->
  {ok, NewState :: tuple()}.

-callback handle()

fire() ->


