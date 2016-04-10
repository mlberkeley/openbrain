-module(test).

-export([makeNopeTimes/0]).

makeNopeTimes() ->
  {echo,java@maxbook} ! {self(),"Hello, Java!"},
  getMessages().

getMessages() ->
  receive
    {connect,Sender} ->
      io:format("Got a message from ~p~n", [Sender]);
    {pixels, PixelData} ->
      io:format("Got pixels~n", [])
    end,
  getMessages().
