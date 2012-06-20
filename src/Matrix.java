import static com.googlecode.javacv.cpp.opencv_core.*;

public class Matrix extends CvMat { 
	
	
	public Matrix Mul(Matrix mat)
	{
		Matrix ret = (Matrix)CvMat.create(this.rows(), mat.cols(),this.depth());	
		cvMatMul(this,mat,ret);
		return ret;
		
	}

}
