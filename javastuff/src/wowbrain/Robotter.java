package wowbrain;

import java.awt.AWTException;
import java.awt.Robot;
import java.awt.event.InputEvent;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.ericsson.otp.erlang.OtpErlangAtom;
import com.ericsson.otp.erlang.OtpErlangDecodeException;
import com.ericsson.otp.erlang.OtpErlangExit;
import com.ericsson.otp.erlang.OtpErlangObject;
import com.ericsson.otp.erlang.OtpErlangPid;
import com.ericsson.otp.erlang.OtpErlangString;
import com.ericsson.otp.erlang.OtpErlangTuple;
import com.ericsson.otp.erlang.OtpMbox;
import com.ericsson.otp.erlang.OtpNode;

public class Robotter {
	protected static OtpNode myOtpNode;
	protected static OtpMbox myOtpMbox;
	protected static OtpErlangPid pixelRecipient;
	
	public static void doHandshake(){
		System.out.println("Starting handshake");
		try {
			myOtpNode = new OtpNode("pxserver");
			System.out.println("OTP Node Started");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		myOtpMbox = myOtpNode.createMbox("pxinbox");
		if (myOtpNode.ping("pixel_register@sus", 2000)) {
			System.out.println("Pinged pixel_register");
		} else {
			System.out.println("couldn't ping");
		}
		
		OtpErlangObject myObject = null;

		OtpErlangTuple myMsg;

		OtpErlangString theMessage;

		OtpErlangAtom myAtom = new OtpErlangAtom("pong");
		
		try {
			System.out.println("trying handshake");
			myObject = myOtpMbox.receive();
			System.out.println("Handshake");
		} catch (OtpErlangExit e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (OtpErlangDecodeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		myMsg = (OtpErlangTuple) myObject;

        pixelRecipient = (OtpErlangPid) myMsg.elementAt(0);

        theMessage = (OtpErlangString) myMsg.elementAt(1);
        System.out.println(theMessage);
        OtpErlangObject[] reply = new OtpErlangObject[2];

        reply[0] = myAtom;

        reply[1] = myOtpMbox.self();

        OtpErlangTuple myTuple = new OtpErlangTuple(reply);

        myOtpMbox.send(pixelRecipient, myTuple);
	}
	
	public static void startGame(){
		Robot robot = null;
		try {
			robot = new Robot();
		} catch (AWTException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		try {
			System.out.println("Running Minecraft");
			Runtime.getRuntime().exec("java -jar /Users/philkuz/Documents/Minecraft.jar");
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		robot.setAutoDelay(8000);
		
		robot.mouseMove(624, 760);
		
		robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);

		robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
		
		robot.setAutoDelay(3500);
		robot.mouseMove(346, 199);
		
		robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
		
		robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
		
		robot.setAutoDelay(3500);
		robot.mouseMove(658, 280);
		
		robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
		
		robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
		
		robot.setAutoDelay(3500);
		robot.mouseMove(606, 54);
		
		robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
		
		robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
	}
	
	public static void delegateJobs(){
	    // start a worker thread
	    ExecutorService exec = Executors.newFixedThreadPool(2);

	    while(true){
		    // ask the worker thread to execute a task (
		    exec.submit(() -> {
		        ScreenGrabberDude.sendScreenPixels();
		    });
		    
		    exec.submit(() -> {
		        ActionDoer.doStuff();
		    });
	    }

	    // terminate the worker thread (otherwise, the thread will wait for more work)
//	    exec.shutdown();
	}

	public static void main(String[] args) {
		doHandshake();
		System.out.println("Handshake done; Starting game.");
		startGame();
	}

}
