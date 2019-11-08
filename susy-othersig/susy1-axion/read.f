      program test_prog
      implicit none
      external read_file

 
      real, dimension (1:275097, 1:1) :: Array1,Array2
      real :: distance(275097)
      integer :: i,j,l
      INTEGER, ALLOCATABLE :: otimalpoint(:)



    
      call read_file (11, 'fpr.txt',275097, 1, Array1)
      call read_file (12, 'tpr.txt',275097, 1, Array2)
      
     
 814      format(2(1pE15.5,2x))


      do i=1,275097
!      OPEN(UNIT=55,FILE='sb.dat',access='append',STATUS ='OLD',
!     $     FORM='FORMATTED')
!      write(55,814)(Arrayp1(i,1)*Array1(i,1))/
!     $(Arrayp2(i,1)*(1-Array2(i,1)))
!      rewind(unit=55) 
      distance(i)=sqrt(Array1(i,1)**2+(Array2(i,1)-1.0)**2)
      print*,"distance",distance(i)
      
      end do
      otimalpoint=minloc(distance)
      print*,"distance",otimalpoint,Array1(135671,1),Array2(135671,1)
      end program test_prog



           

        

      subroutine read_file(UnitNum, FileName, NumRows, NumCols, Array)
      implicit none
      integer, intent (in) :: UnitNum
      character (len=*), intent (in) :: FileName
      integer, intent (in) :: NumRows, NumCols
      real, dimension (1:NumRows, 1:NumCols), intent (out) :: Array
      integer :: i, j

      open (unit=UnitNum, file=FileName, status='old')
      do i=1, NumRows
      read (UnitNum, *) (Array (i, j), j=1,NumCols)
      end do
      close (UnitNum)
      return
      end subroutine


