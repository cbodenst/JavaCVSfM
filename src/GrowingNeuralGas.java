import static com.googlecode.javacv.cpp.opencv_core.*;

import java.util.ArrayList;

public class GrowingNeuralGas extends Thread {
	private int idpool=0;
	
	private final double lambda=20;
	private final double e_w=0.5;
	private final double e_n=0.05;
	private final int A_MAX =5;
	private final double alpha = 0.5;
	private final double beta = 0.0005;
	private final int max_nodes=40;
	

	private Node s;
	private Node t;
	private Node r;
	private Node u;
	private Node v;
	private int counter;
	

	private ArrayList<Node> nodes = new ArrayList<Node>();
	
	public GrowingNeuralGas()
	{
		v = new Node();
		u= new Node();
		Edge e = new Edge(v,u,0);
		v.edges.add(e);
		v.weight.put(0,0,44);
		v.weight.put(1,0,0);
		v.weight.put(2,0,24);
		u.edges.add(e);

		u.weight.put(0,0,44);
		u.weight.put(1,0,6);
		u.weight.put(2,0,24);
			
		nodes.add(v);
		nodes.add(u);
	}
	
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
			Node otherNode = e.getConnectedNode(this);
			edges.remove(e);
			otherNode.edges.remove(e);
			if(this.edges.size()<=0)
			{
				nodes.remove(this);
				System.out.println("NODE REMOVED");
			}
			if(otherNode.edges.size()<=0)
			{
				nodes.remove(otherNode);		
				System.out.println("NODE REMOVED");
			}

		}
	}
	
	private void removeEdges()
	{
		for (int i = 0;i<nodes.size();i++)
		{
			Node n = nodes.get(i);
			for(int j=0; j<n.edges.size();j++)
			{
				Edge e = n.edges.get(j);
				if(e.age > A_MAX)
				{
					s.removeEdge(e);
				}
			}
		}
	}
	
	private double norm(Node n,CvMat x)
	{
		double value = Math.pow(x.get(0,0)-n.weight.get(0,0),2);
		value += Math.pow(x.get(1,0)-n.weight.get(1,0),2);
		value += Math.pow(x.get(2,0)-n.weight.get(2,0),2);
		
		return value;
		
	}
	
	private void findnearestsNeighbours(CvMat x)
	{
		Node next = nodes.get(0);
		double nextvalue = norm(next,x);
		Node nextnext = nodes.get(1);
		for (int i = 0;i<nodes.size();i++)
		{
			Node n = nodes.get(i);
			double value = norm(n,x);
			if(value < nextvalue)
			{
				nextnext=next;
				
				next=n;
				nextvalue=value;
			}
		}
		s=next;
		t=nextnext;
		s.error+=nextvalue;
	}
	private void updateWeights(CvMat x)
	{
		s.weight.put(0,0,s.weight.get(0,0)+e_w*(x.get(0,0)-s.weight.get(0,0)));
		s.weight.put(1,0,s.weight.get(1,0)+e_w*(x.get(1,0)-s.weight.get(1,0)));
		s.weight.put(2,0,s.weight.get(2,0)+e_w*(x.get(2,0)-s.weight.get(2,0)));
		
		boolean t_found = false;
		
		for(int i = 0; i<s.edges.size();i++)
		{
			Edge e = s.edges.get(i);
				
			Node n = e.getConnectedNode(s);
			n.weight.put(0,0, n.weight.get(0,0) + e_n*( x.get(0,0)-n.weight.get(0,0) ) );
			n.weight.put(1,0, n.weight.get(1,0) + e_n*( x.get(1,0)-n.weight.get(1,0) ) );
			n.weight.put(2,0, n.weight.get(2,0) + e_n*( x.get(2,0)-n.weight.get(2,0) ) );
			
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
	
	private void addNode()
	{
		double max = -1;
		for(int i = 0; i<nodes.size();i++)
		{
			Node n = nodes.get(i);
			if(s.error>max)
			{
				max=s.error;
				u=n;
			}
		}
		max=-1;
		Edge todelete=null;
		for (int i = 0; i < u.edges.size();i++)
		{
			Edge e = u.edges.get(i);
			Node n = e.getConnectedNode(u);
			double value = n.error;
			if(value>max)
			{
				max=value;
				v=n;
				todelete=e;
			}
		}
		r = new Node();
		r.weight.put(0,0,(u.weight.get(0,0)+v.weight.get(0,0))/2);
		r.weight.put(1,0,(u.weight.get(1,0)+v.weight.get(1,0))/2);
		r.weight.put(2,0,(u.weight.get(2,0)+v.weight.get(2,0))/2);
		
		Edge e1 =new Edge(u,r,0);
		Edge e2 = new Edge(v,r,0);
			
			
		r.edges.add(e1);
		r.edges.add(e2);
		u.edges.add(e1);
		v.edges.add(e2);
		

		u.removeEdge(todelete);
		
		nodes.add(r);
			
		u.error *= alpha;
		v.error *= alpha;
		r.error=u.error;
			
			
			

	}
	
	private void recalculateErrors()
	{
		for (int i = 0;i<nodes.size();i++)
		{
			Node n = nodes.get(i);
			n.error -= n.error*beta;
		}
	}
	
	public void printNodes()
	{
		for (int i = 0;i<nodes.size();i++)
		{
			Node n = nodes.get(i);
			System.out.println(n.weight.get(0,0)+"\t"+n.weight.get(1,0)+"\t"+n.weight.get(2,0));
		}
	}
	public void input(CvMat x)
	{
		counter++;
		findnearestsNeighbours(x);
		updateWeights(x);
		removeEdges();
		if(counter>lambda)
		{
			counter = 0;
			if(nodes.size()<max_nodes)
				addNode();
		}
		recalculateErrors();
		printError();
	}
	
	public void printError()
	{
		double mean_error=0;
		for (int i = 0;i<nodes.size();i++)
		{
			Node n = nodes.get(i);
			mean_error+=n.error;
		}
		System.out.println("Mean-Error:"+mean_error/(double)nodes.size());
		
	}
	
	
	public void run()
	{
	}

}
