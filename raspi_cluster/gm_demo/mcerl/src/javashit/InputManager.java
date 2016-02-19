package javashit;

import java.awt.AWTException;
import java.awt.MouseInfo;
import java.awt.Point;
import java.awt.Robot;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import com.ericsson.otp.erlang.OtpErlangAtom;
import com.ericsson.otp.erlang.OtpErlangBinary;
import com.ericsson.otp.erlang.OtpErlangDecodeException;
import com.ericsson.otp.erlang.OtpErlangExit;
import com.ericsson.otp.erlang.OtpErlangObject;
import com.ericsson.otp.erlang.OtpErlangPid;
import com.ericsson.otp.erlang.OtpErlangTuple;
import com.ericsson.otp.erlang.OtpMbox;
import com.ericsson.otp.erlang.OtpNode;

public class InputManager implements Runnable, KeyListener{
	private static final byte W = 0b1000000;
	private static final byte A = 0b0100000;
	private static final byte S = 0b0010000;
	private static final byte D = 0b0001000;
	private static final byte LEFT = 0b0000100;
	private static final byte RIGHT = 0b0000010;
	private static final byte CLICK = 0b0000001;
	private static int XSHIFT = 100;
	boolean enabled = false;
	int dir;
	private Robot mrRoboto;
	private Point p;
	
	/**
	 * Sets the internal connection and assumes 
	 * EC has already been connected.
	 * @param ec
	 */
	public InputManager(){
		try {
			this.mrRoboto = new Robot();
			this.p = new Point(0,0);
		} catch (AWTException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	@Override
	public void run() {
		try {
			while(true){
				if(!enabled){
					try {
						System.out.println("asstitties");
						Thread.sleep(3000);
		        		p = MouseInfo.getPointerInfo().getLocation();
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
				else{
					getInput();
				}
			}
		} catch (OtpErlangExit | OtpErlangDecodeException | AWTException e) {
			e.printStackTrace();
		}
	}

	private void getInput() throws AWTException, OtpErlangExit, OtpErlangDecodeException {
		OtpErlangBinary reply = (OtpErlangBinary) Program.javaMbox.receive();

		byte[] bytes = reply.binaryValue();

		byte command = bytes[0];

		System.out.println(reply.toString());

		mrRoboto.keyRelease(KeyEvent.VK_W);
		mrRoboto.keyRelease(KeyEvent.VK_A);
		mrRoboto.keyRelease(KeyEvent.VK_S);
		mrRoboto.keyRelease(KeyEvent.VK_D);
		mrRoboto.keyRelease(KeyEvent.VK_SPACE);
		 
		
		if((W & command) != 0){
			mrRoboto.keyPress(KeyEvent.VK_W);
			System.out.println("w");
		}
		if((A & command) != 0){
			mrRoboto.keyPress(KeyEvent.VK_A);
		}
		if((S & command) != 0){
			mrRoboto.keyPress(KeyEvent.VK_S);
		}
		if((D & command) != 0){
			mrRoboto.keyPress(KeyEvent.VK_D);
		}
		if((LEFT & command) != 0){
			dir = -1;
			System.out.println(dir);
    		int x = (int) (p.getX()) ;
    		int y = (int) p.getY();
			mrRoboto.mouseMove(x+ dir*XSHIFT,y);
		}
		else if((RIGHT & command) != 0){

			mrRoboto.keyPress(KeyEvent.VK_SPACE);
			
		}
		else
			dir = 0;
		if((CLICK & command) != 0){
			int button = InputEvent.BUTTON1_DOWN_MASK;
			mrRoboto.mousePress(button);
			mrRoboto.mouseRelease(button);
		}
	
	}

	@Override
	public void keyTyped(KeyEvent e) {
		System.out.println("om nom");
		if(e.getKeyChar() == 'a'){
			System.out.println("eyyy");
			enabled = !enabled;
		}
	}

	@Override
	public void keyPressed(KeyEvent e) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void keyReleased(KeyEvent e) {
		// TODO Auto-generated method stub
		
	}

}
