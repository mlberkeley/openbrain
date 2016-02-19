-module(test).

-export([send/1]).


send(Host) ->
  {inputListener, 'com@Maxs-MacBook-Pro-2.local'} ! Host,
  timer:sleep(1000),
  send(Host).
