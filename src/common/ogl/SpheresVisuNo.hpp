#ifndef OGL_SPHERES_VISU_NO_HPP_
#define OGL_SPHERES_VISU_NO_HPP_

#include "SpheresVisu.hpp"

template <typename T> class SpheresVisuNo : public SpheresVisu {
  public:
    SpheresVisuNo();

    virtual ~SpheresVisuNo();

    void refreshDisplay();
    bool windowShouldClose();
    bool pressedSpaceBar();
    bool pressedPageUp();
    bool pressedPageDown();
};

#endif /* OGL_SPHERES_VISU_NO_HPP_ */
