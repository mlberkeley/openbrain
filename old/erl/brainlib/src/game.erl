-module(game).

-export([start/0, erl_master/0, pixels/1, start_input/0, input/0, stop/0]).


erl_master() ->
  receive
    {setup_java, JavaPid} ->
      io:format("Received Java input ~w~n", [JavaPid]),
      PixelPid = spawn(game, pixels, [brain]),
      JavaPid ! {pixelpid, PixelPid},
      register(input_listener, spawn(game, input, []))
    end.

pixels(NNPid) ->
  receive
    %send pixels to neural network.
    {pixels, Pixels} ->
      NNPid ! {feed, Pixels},

      pixels(NNPid)
    end.

start_input()->
  receive
    {connect, Pid} ->
      io:format("~p connected to input listener!~n", [Pid]),
      input()
    end.

input() ->
    receive
      w ->
        {actioninbox, 'actionserver@maxbook'} ! w,
        input();
      a ->
        {actioninbox, 'actionserver@maxbook'} ! a,
        input();
      s ->
        {actioninbox, 'actionserver@maxbook'} ! s,
        input();
      d ->
        {actioninbox, 'actionserver@maxbook'} ! d,
        input();
      space ->
        {actioninbox, 'actionserver@maxbook'} ! space,
        input();
      mouse ->
        {actioninbox, 'actionserver@maxbook'} ! mouse,
        input()
      end.

start() ->
    brain:start(),
    ErlMaster = spawn(game, erl_master, []),
    {pxinbox, 'pxserver@maxbook'} ! {ErlMaster, "Hello, Java!"}.



stop() ->
     unregister(pixel_register),
     unregister(input_listener).
