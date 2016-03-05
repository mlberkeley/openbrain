%% @author philkuz
%% @doc @todo Add description to mcdemo.


-module(mcdemo).

%% ====================================================================
%% API functions
%% ====================================================================
-export([start/0,start/2, stop/0]).
start() ->
	start(normal, placeholder).
%% @doc Starts the module
start(normal, _Args) ->
	brain:start(),
	game:start().
%% @doc Stops the module
stop() ->
	game:stop(),
	brain ! stop.
	
%% ====================================================================
%% Internal functions
%% ====================================================================


