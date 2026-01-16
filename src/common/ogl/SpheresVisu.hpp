#ifndef SPHERES_VISU_HPP_
#define SPHERES_VISU_HPP_

class SpheresVisu {
  protected:
    SpheresVisu() {}

  public:
    virtual ~SpheresVisu() {}
    virtual void refreshDisplay() = 0;
    virtual bool windowShouldClose() = 0;
    virtual bool pressedSpaceBar() = 0;
    virtual bool pressedPageUp() = 0;
    virtual bool pressedPageDown() = 0;
};

#endif /* SPHERES_VISU_HPP_ */
