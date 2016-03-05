-module(game).

-export([start/0, erl_master/0, pixels/1, start_input/0, input/1, stop/0]).


erl_master() ->
  receive
%% make this fault tolerant
    {setup_java, JavaPid} ->
      io:format("Received Java input ~w~n", [JavaPid]),
      PixelPid = spawn(game, pixels, [brain]),
      JavaPid ! {pixelpid, PixelPid},
      register(input_listener, spawn(game, input, [JavaPid]))
%%    {javapid, Pid} ->
%%      io:format("~p connected to pixel listener!~n", [Pid]),
%%      register(pixel_listener, spawn(game, pixels, [Pid]));
%%    {getpid, JavaPid} ->
%%      PixelPid = spawn(game, pixels, [brain]),
%%      JavaPid ! PixelPid
    end.

pixels(NNPid) ->
  receive
    %send pixels to neural network.
    {pixels, Pixels} ->
      NNPid ! {feed, Pixels},

      pixels(NNPid)
	
%%     Other ->
%%       unregister(pixel_listener),
%%       register(pixel_listener, spawn(game, erl_master, [])),
%%       pixel_listener ! Other
    end.

start_input()->
  receive
    {connect, Pid} ->
      io:format("~p connected to input listener!~n", [Pid]),
      input(Pid)
%%      Other ->
%%        io:format("shit~n"),
%%        stop()
    end.

input(JavaPid) ->
    receive
      w ->
%%        io:format('pressed w ~n', []),
        JavaPid ! w,
        input(JavaPid);
      a ->
%%        io:format('pressed a ~n', []),
        JavaPid ! a,
        input(JavaPid);
      s ->
        JavaPid ! s,
        input(JavaPid);
      d ->
        JavaPid ! d,
        input(JavaPid);
%%       left ->
%%         JavaPid ! <<4>>,
%%         input(JavaPid);
      space ->
        JavaPid ! space,
        input(JavaPid);
      mouse ->
        JavaPid ! mouse,
        input(JavaPid)
%%       Other ->
%%         unregister(input_listener),
%%         register(input_listener, spawn(game, start_input, [])),
%%         input_listener ! Other
      end.

start() ->
    brain:start(),
    ErlMaster = spawn(game, erl_master, []),
%%    register(pixel_register, PixelPid),
    {pxinbox, 'pxserver@maxbook'} ! {ErlMaster, "Hello, Java!"}.
%%  try {pxinbox, phillipMBP} ! {PixelPid, "Hello, Java!"} of
%%      _ -> ok
%%  catch
%%      _:Type  ->  io:format("Couldn't connect for ~w ~n", [Type])
%%  end ,
%%    pxinbox ! {PixelPid, "Hello, Java!"},


stop() ->
     unregister(pixel_register),
     unregister(input_listener).
