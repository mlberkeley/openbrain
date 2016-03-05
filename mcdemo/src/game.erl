-module(game).

-export([start/0, start_pixels/0, pixels/1, start_input/0, input/1, stop/0]).


start_pixels() ->
  receive
%% make this fault tolerant 	  
    Pid ->
      io:format("~p connected to pixel listener!~n", [Pid]),
      register(pixel_listener, spawn(game, pixels, [Pid]))
    end.

pixels(NNPid) ->
  receive
    %send pixels to neural network.
    {pixels, Pixels} ->
      %NNPid ! Pixels

      pixels(NNPid)
	
%%     Other ->
%%       unregister(pixel_listener),
%%       register(pixel_listener, spawn(game, start_pixels, [])),
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
        io:format('pressed w ~n', []),
        JavaPid ! w,
        input(JavaPid);
      a ->
        io:format('pressed a ~n', []),
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
        JavaPid ! <<2>>,
        input(JavaPid);
      mouse ->
        JavaPid ! <<1>>,
        input(JavaPid)
%%       Other ->
%%         unregister(input_listener),
%%         register(input_listener, spawn(game, start_input, [])),
%%         input_listener ! Other
      end.

start() ->
    PixelPid = spawn(game, start_pixels, []),
    register(pixel_register, PixelPid),
%%    pxserver ! {PixelPid, "Hello, Java!"},
%%  try {pxinbox, phillipMBP} ! {PixelPid, "Hello, Java!"} of
%%      _ -> ok
%%  catch
%%      _:Type  ->  io:format("Couldn't connect for ~w ~n", [Type])
%%  end ,
%%    pxinbox ! {PixelPid, "Hello, Java!"},
    register(input_listener, spawn(game, start_input, [])).

stop() ->
     unregister(pixel_register),
     unregister(input_listener).
