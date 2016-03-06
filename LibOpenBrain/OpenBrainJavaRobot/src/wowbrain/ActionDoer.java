package wowbrain;

import com.ericsson.otp.erlang.*;

import java.awt.*;
import java.awt.event.KeyEvent;
import java.io.IOException;

public class ActionDoer {
	public static void doStuff(){
		OtpNode actionOtpNode = null;
		OtpMbox actionMbox = null;

		try {
			actionOtpNode = new OtpNode("actionserver");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		actionMbox = actionOtpNode.createMbox("actioninbox");

		while(true){
			try{
				Thread.sleep(100);

			} catch (InterruptedException e){
			}

			Robot robot = null;
			try {
				robot = new Robot();
			} catch (AWTException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			OtpErlangAtom theMessage = null;

			OtpErlangAtom myAtom = new OtpErlangAtom("pong");

			System.out.println("getting command from erlang");

			try {
				theMessage = (OtpErlangAtom) actionMbox.receive();
			} catch (OtpErlangExit e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (OtpErlangDecodeException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			System.out.println("got " + theMessage);

			if(Robotter.enabled){
				int rand_factor = 5 * 200;
				if(theMessage.toString().equals("w")){
					robot.keyPress(KeyEvent.VK_W);
					robot.delay((int) (Math.random() * rand_factor));
					robot.keyRelease(KeyEvent.VK_W);
				} else if(theMessage.toString().equals("a")){
					robot.keyPress(KeyEvent.VK_A);
					robot.delay((int) (Math.random() * rand_factor));
					robot.keyRelease(KeyEvent.VK_A);
				} else if(theMessage.toString().equals("s")){
					robot.keyPress(KeyEvent.VK_S);
					robot.delay((int) (Math.random() * rand_factor));
					robot.keyRelease(KeyEvent.VK_S);
				} else if(theMessage.toString().equals("d")){
					robot.keyPress(KeyEvent.VK_D);
					robot.delay((int) (Math.random() * rand_factor));
					robot.keyRelease(KeyEvent.VK_D);
				}
			}
		}
	}
}
