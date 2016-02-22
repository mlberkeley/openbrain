package javastuff;

import java.io.IOException;

import com.ericsson.otp.erlang.OtpErlangAtom;
import com.ericsson.otp.erlang.OtpErlangDecodeException;
import com.ericsson.otp.erlang.OtpErlangExit;
import com.ericsson.otp.erlang.OtpErlangObject;
import com.ericsson.otp.erlang.OtpErlangPid;
import com.ericsson.otp.erlang.OtpErlangTuple;
import com.ericsson.otp.erlang.OtpMbox;
import com.ericsson.otp.erlang.OtpNode;

/**
 *
 */
public class Program {
	static OtpNode javaNode;
	static OtpMbox javaMbox;
	static final String erlNodeName = "erlNode";
	static OtpErlangPid erlNodePid;

	public static void initConnection() throws IOException, OtpErlangExit, OtpErlangDecodeException{
		javaNode = new OtpNode("java");
		javaMbox = javaNode.createMbox("echo");
		
		System.out.println("Listening for initial ping");

		OtpErlangObject message = javaMbox.receive();
		System.out.format("%s received: %s%n", javaMbox.self(), message);
		
		OtpErlangTuple messageTuple = (OtpErlangTuple) message;
		
		erlNodePid = (OtpErlangPid) messageTuple.elementAt(0);
		
		OtpErlangObject[] msg = new OtpErlangObject[2];
		msg[0] = new OtpErlangAtom("connect");
		msg[1] = javaMbox.self();
		OtpErlangTuple tuple = new OtpErlangTuple(msg); 
        javaMbox.send(erlNodePid, tuple);
	}
	
	public static void main(String[] args) throws IOException, OtpErlangExit, OtpErlangDecodeException {
		initConnection();
		
		InputManager im = new InputManager();
		
		Thread input = new Thread(im);
		
		input.start();
		
		PixelManager pm = new PixelManager();
		Thread pixel = new Thread(pm);
		pixel.start();
		pm.getFrame().addKeyListener(im);
		
//		try {
//			OtpNode localNode = new OtpNode("binterface@Maxs-MacBook-Pro-2.local");
//			InputManager im = new InputManager(localNode, nodeName);
//			Thread input = new Thread(im);
//			input.start();
//			
//			PixelManager pm = new PixelManager(localNode, nodeName);
//			Thread pixel = new Thread(pm);
//			pixel.start();
//			pm.getFrame().addKeyListener(im);
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
	}
}