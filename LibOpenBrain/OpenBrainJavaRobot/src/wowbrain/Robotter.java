package wowbrain;

import com.ericsson.otp.erlang.*;
import javax.swing.*;
import java.awt.*;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Robotter {
	protected static OtpNode myOtpNode;
	protected static OtpMbox myOtpMbox;
	protected static OtpErlangPid erl_master;
	protected static OtpErlangPid pixel_pid;
	protected static boolean enabled = true;

	
	public static void doHandshake(){
		try {
			myOtpNode = new OtpNode("pxserver");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		myOtpMbox = myOtpNode.createMbox("pxinbox");

		OtpErlangObject myObject = null;

		OtpErlangTuple myMsg;

		OtpErlangString theMessage;
		
		try {
			myObject = myOtpMbox.receive();
		} catch (OtpErlangExit e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (OtpErlangDecodeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		myMsg = (OtpErlangTuple) myObject;

        if(myMsg != null){
            erl_master = (OtpErlangPid) myMsg.elementAt(0);
            theMessage = (OtpErlangString) myMsg.elementAt(1);
        } else {
            throw new RuntimeException("Didn't get a message");
        }


        System.out.println(theMessage);
        
        OtpErlangObject[] reply = new OtpErlangObject[2];

        reply[0] = new OtpErlangAtom("setup_java");

        reply[1] = myOtpMbox.self();

        OtpErlangTuple myTuple = new OtpErlangTuple(reply);

        myOtpMbox.send(erl_master, myTuple);
        
        OtpErlangObject pixelObject = null;
		try {
			pixelObject = myOtpMbox.receive();
		} catch (OtpErlangExit e) {
			e.printStackTrace();
		} catch (OtpErlangDecodeException e) {
			e.printStackTrace();
		}
		
		OtpErlangTuple pixelMessage = (OtpErlangTuple) pixelObject;

        if(pixelMessage != null){
            pixel_pid = (OtpErlangPid) pixelMessage.elementAt(1);
            System.out.println(pixelMessage.elementAt(0));
        } else {
            throw new RuntimeException("Malformed handshake response");
        }
	}
	
	public static void startGame(){
		Robot robot = null;
		try {
			robot = new Robot();
		} catch (AWTException e1) {
			e1.printStackTrace();
		}
		
		try {
			Runtime.getRuntime().exec("java -jar /Users/maxjohansen/Minecraft.jar");
		} catch (IOException e1) {
			e1.printStackTrace();
		}

        if(robot == null){
            throw new RuntimeException("Couldn't get robot");
        }

		robot.delay(8000);
		
		robot.mouseMove(624, 760);
		
		robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);

		robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
		
		robot.delay(3500);
		robot.mouseMove(346, 199);
		
		robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
		
		robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
		
		robot.delay(3500);
		robot.mouseMove(658, 280);
		
		robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
		
		robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
		
		robot.delay(3500);
		robot.mouseMove(606, 54);
		
		robot.mousePress(InputEvent.BUTTON1_DOWN_MASK);
		
		robot.mouseRelease(InputEvent.BUTTON1_DOWN_MASK);
	}
	
	public static void delegateJobs(){
	    // start a worker thread
	    ExecutorService exec = Executors.newFixedThreadPool(2);
		// ask the worker thread to execute a task
		exec.submit(new Runnable() {
			@Override
			public void run() {
				ScreenGrabberDude.sendScreenPixels();
			}
		});

		exec.submit(new Runnable() {
			@Override
			public void run() {
				ActionDoer.doStuff();
			}
		});

		// terminate the worker thread (otherwise, the thread will wait for more work)
	    exec.shutdown();
	}

	public static void main(String[] args) {
		doHandshake();
		
//		startGame();

		JFrame frame = new JFrame();
		frame.getContentPane().setLayout(new FlowLayout());

		frame.pack();
		frame.setVisible(true);

		frame.addKeyListener(new KeyListener() {
			@Override
			public void keyTyped(KeyEvent e) {

			}

			@Override
			public void keyPressed(KeyEvent e) {
				if(e.getKeyChar() == 'q'){
					System.out.println("toggle");
					enabled = !enabled;
				}
			}

			@Override
			public void keyReleased(KeyEvent e) {

			}
		});

//		JLabel img = new JLabel(new ImageIcon(capture()));
//		frame.getContentPane().add(img);

		delegateJobs();
	}

}
