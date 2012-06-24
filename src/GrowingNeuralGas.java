import static com.googlecode.javacv.cpp.opencv_core.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

public class GrowingNeuralGas extends Thread {
	private int idpool=0;
	private double lambda=0.5;
	private Node s;
	private Node t;
	private Node r;
	private Node u;
	private Node v;
	private final double e_w=0.5;
	private final double e_n=0.5;
	private final int A_MAX =50;
	
	private class Edge
	{
		public int age;
		public Node node1;
		public Node node2;
		
		public Edge(Node n1,Node n2,int age)
		{
			this.node1=n1;
			this.node2=n2;
			this.age=age;
		}
		
		public Node getConnectedNode(Node n)
		{
			if (n.id == node1.id)
				return node2;
			return node1;
		}
	}
	
	
	
	private class Node
	{
		public int id;
		public double error;
		public CvMat weight;
		public ArrayList<Edge> edges;
		
		public Node()
		{
			this.id = idpool++;
			this.error = 0;
			this.weight =  CvMat.create(3,1,CV_32F);
			this.edges = new ArrayList<Edge>();
		}
		
		public void removeEdge(Edge e)
		{
			edges.remove(e);
			Node otherNode = nodes.get(e.getConnectedNode(this));
			otherNode.removeEdge(e);
			if(this.edges.size()<=0)
				nodes.remove(this);
			if(otherNode.edges.size()<=0)
				nodes.remove(otherNode);
		}
	}
	private HashMap<Integer,Node> nodes = new HashMap<Integer,Node>();
	
	private void findnearestsNeighbours(CvMat x)
	{
		double max=0;
		Node next = null;
		double nextvalue = 1000000000;
		Node nextnext = null;
		double nextnextvalue = 1000000000;
		Iterator it = nodes.entrySet().iterator();
		while(it.hasNext())
		{
			Node n = (Node)it.next();
			double value = Math.pow(x.get(0,0)-n.weight.get(0,0),2);
			value += Math.pow(x.get(1,0)-n.weight.get(1,0),2);
			value += Math.pow(x.get(2,0)-n.weight.get(2,0),2);
			if(value < nextvalue)
			{
				nextnext=next;
				nextnextvalue=nextvalue;
				next=n;
				nextvalue=value;
			}
			if(value>max)
			{
				max=value;
				u=n;
			}
			s=next;
			t=nextnext;
			s.error+=nextvalue;
			t.error+=nextnextvalue;
		}
	}
	private void updateWeights(CvMat x)
	{
		s.weight.put(0,0,s.weight.get(0,0)+e_w*(s.weight.get(0,0)-x.get(0,0)));
		s.weight.put(1,0,s.weight.get(1,0)+e_w*(s.weight.get(1,0)-x.get(1,0)));
		s.weight.put(2,0,s.weight.get(2,0)+e_w*(s.weight.get(2,0)-x.get(2,0)));
		
		boolean t_found = false;
		
		for(int i = 0; i<s.edges.size();i++)
		{
			Edge e = s.edges.get(i);
			if(e.age>A_MAX)
			{
				s.removeEdge(e);
				break;
			}
				
			Node n = e.getConnectedNode(s);
			n.weight.put(0,0,n.weight.get(0,0)+e_n*(n.weight.get(0,0)-x.get(0,0)));
			n.weight.put(1,0,n.weight.get(1,0)+e_n*(n.weight.get(1,0)-x.get(1,0)));
			n.weight.put(2,0,n.weight.get(2,0)+e_n*(n.weight.get(2,0)-x.get(2,0)));
			
			e.age++;
			
			if(n.id == t.id)
			{
				t_found=true;
				e.age=0;
			}
			
		}
		
		if(!t_found)
		{
			Edge e = new Edge(s,t,0);
			s.edges.add(e);
			t.edges.add(e);			
		}
	}
	
	private void addNode(CvMat x)
	{
		double max=0;
		Iterator it = u.edges.iterator();
		while(it.hasNext())
		{
			Edge e = (Edge)it.next();
			Node n = e.getConnectedNode(u);
			double value = n.error;
			if(value>max)
			{
				max=value;
				v=n;
			}
			Node r = new Node();
			r.weight.put(0,0,(u.weight.get(0,0)+v.weight.get(0,0))/2);
			r.weight.put(1,0,(u.weight.get(1,0)+v.weight.get(1,0))/2);
			r.weight.put(2,0,(u.weight.get(2,0)+v.weight.get(2,0))/2);
			
			Edge e1 =new Edge(u,r,0);
			Edge e2 = new Edge(v,r,0);
			
			r.edges.add(e1);
			r.edges.add(e2);
			u.edges.add(e1);
			v.edges.add(e2);
			
			
		}	
		
	}
	
	
	public void run()
	{
	}

}
