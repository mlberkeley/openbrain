-module(game).

-export([start/0, startPixels/0, pixels/1, startInput/0, input/1, stop/0]).


startPixels() ->
  receive
    {pixels, Pixels} ->
      startPixels(); % still waiting for output loz
    Pid ->
        io:format("~p connected to pixel listener!~n", [Pid]),
        pixels(Pid)
    end.

pixels(NNPid) ->
  receive
    %send pixels to neural network.
    {pixels, Pixels} ->
      %NNPid ! Pixels

      pixels(NNPid);
    Other ->
      unregister(pixelListener),
      register(pixelListener, spawn(game, startPixels, [])),
      pixelListener ! Other
    end.

startInput()->
  receive
    {connect, Pid} ->
      io:format("~p connected to input listener!~n", [Pid]),
      input(Pid);
     Other ->
       io:format("shit~n"),
       stop()
    end.

input(JavaPid) ->
    receive
      w ->
        io:format('pressed w ~n', []),
        JavaPid ! <<64>>,
        input(JavaPid);
      a ->
        io:format('pressed a ~n', []),
        JavaPid ! <<32>>,
        input(JavaPid);
      s ->
        JavaPid ! <<16>>,
        input(JavaPid);
      d ->
        JavaPid ! <<8>>,
        input(JavaPid);
      left ->
        JavaPid ! <<4>>,
        input(JavaPid);
      space ->
        JavaPid ! <<2>>,
        input(JavaPid);
      mouse ->
        JavaPid ! <<1>>,
        input(JavaPid);
      Other ->
        unregister(inputListener),
        register(inputListener, spawn(game, startInput, [])),
        inputListener ! Other
      end.

start() ->
    PixelPid = spawn(game, startPixels, []),
    register(pixelListener, PixelPid),
    {echo,java@maxbook} ! {PixelPid, "Hello, Java!"},
    register(inputListener, spawn(game, startInput, [])).
    

stop() ->
     unregister(pixelListener),
     unregister(inputListener).
