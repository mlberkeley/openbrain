package javastuff;

import java.io.IOException;

import com.ericsson.otp.erlang.OtpErlangAtom;
import com.ericsson.otp.erlang.OtpErlangDecodeException;
import com.ericsson.otp.erlang.OtpErlangExit;
import com.ericsson.otp.erlang.OtpErlangObject;
import com.ericsson.otp.erlang.OtpErlangTuple;
import com.ericsson.otp.erlang.OtpMbox;
import com.ericsson.otp.erlang.OtpNode;

public class TestRx {
	static OtpNode node;
	static OtpMbox mbox;

	public static void main(String[] args) throws IOException, OtpErlangExit, OtpErlangDecodeException  {
		node = new OtpNode("java");
		mbox = node.createMbox("echo");
//		sendShit();

		getShit();
	}

	public static void sendShit(){
		OtpErlangObject[] msg = new OtpErlangObject[2];
		msg[0] = mbox.self();
		msg[1] = new OtpErlangAtom("hello");
		OtpErlangTuple tuple = new OtpErlangTuple(msg); 
		
        mbox.send("erlNode", "erlNode@airbears2-10-142-58-235", tuple);
	}

	public static void getShit() throws IOException, OtpErlangExit, OtpErlangDecodeException{
		System.out.println("Listening for messages");
		while(true){
			OtpErlangObject message = mbox.receive();
			System.out.format("%s received: %s%n", mbox.self(), message);
		}
	}

}
