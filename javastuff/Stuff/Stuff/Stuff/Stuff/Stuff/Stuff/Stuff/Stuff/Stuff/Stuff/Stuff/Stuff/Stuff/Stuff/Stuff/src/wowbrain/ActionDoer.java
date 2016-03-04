package wowbrain;

import java.awt.AWTException;
import java.awt.Robot;
import java.awt.event.KeyEvent;

import com.ericsson.otp.erlang.OtpErlangAtom;
import com.ericsson.otp.erlang.OtpErlangDecodeException;
import com.ericsson.otp.erlang.OtpErlangExit;
import com.ericsson.otp.erlang.OtpErlangObject;
import com.ericsson.otp.erlang.OtpErlangPid;
import com.ericsson.otp.erlang.OtpErlangString;
import com.ericsson.otp.erlang.OtpErlangTuple;

public class ActionDoer {
	public static void doStuff(){
		Robot robot = null;
		try {
			robot = new Robot();
		} catch (AWTException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		OtpErlangObject myObject = null;

		OtpErlangTuple myMsg;

		OtpErlangAtom theMessage;

		OtpErlangAtom myAtom = new OtpErlangAtom("pong");
		
		try {
			myObject = Robotter.myOtpMbox.receive();
		} catch (OtpErlangExit e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (OtpErlangDecodeException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		myMsg = (OtpErlangTuple) myObject;

        theMessage = (OtpErlangAtom) myMsg.elementAt(0);
        
        if(theMessage.toString().equals("w")){
        	robot.keyPress(KeyEvent.VK_W);
        	robot.keyRelease(KeyEvent.VK_W);
        } else if(theMessage.toString().equals("a")){
        	robot.keyPress(KeyEvent.VK_A);
        	robot.keyRelease(KeyEvent.VK_A);
        } else if(theMessage.toString().equals("s")){
        	robot.keyPress(KeyEvent.VK_S);
        	robot.keyRelease(KeyEvent.VK_S);
        } else if(theMessage.toString().equals("d")){
        	robot.keyPress(KeyEvent.VK_D);
        	robot.keyRelease(KeyEvent.VK_D);
        }
	}
}
