%% @author philkuz
%% @doc @todo Add description to utils.

%% @description Handles all of that math utility stuff necessary for 
%% the project.  
-module(extra_math).

%% ====================================================================
%% API functions
%% ====================================================================
-export([floor/1, ceiling/1, mod/2]).

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

%% ====================================================================
%% Internal functions
%% ====================================================================


