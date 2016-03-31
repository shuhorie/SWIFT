/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2016 Tom Theuns (tom.theuns@durham.ac.uk)
 *                    Matthieu Schaller (matthieu.schaller@durham.ac.uk)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

/* Config parameters. */
#include "../config.h"

/* This object's header. */
#include "physical_constants.h"

/* Local headers. */
#include "physical_constants_cgs.h"

/**
 * @brief Converts physical constants to the internal unit system
 *
 * @param us The current internal system of units.
 * @param internal_const The physical constants to initialize.
 */
void initPhysicalConstants(struct UnitSystem* us,
                           struct phys_const* internal_const) {

  const float dimension[5] = {1, -3, 2, 0, 0};
  internal_const->newton_gravity =
      NEWTON_GRAVITY_CGS * generalConversionFactor(us, dimension);
}
