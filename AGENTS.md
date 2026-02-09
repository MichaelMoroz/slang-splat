# AGENTS Code Style Guide
* Before proceeding with editing code, provide an extensive edit plan explaning what and why, which algorithms are you going to use, etc.
* Buffer creation, shader creation, and shader execution + memory readbacks and upload should be in separate methods in python for separation of responsibility. 
* Slangpy uses threadcount for `.dispatch(...)` calls! But it does use workgroup count for indirect dispatches.
* If there is any small vector or matrix data - use spy vectors (like `spy.float3`) or spy matrices (like `spy.float3x3`). That includes things like spherical coordinates, deltas, mouse positions, camera matrices, projection matrices, etc.
* Avoid using per-component operations on vectors, if something is 2d, 3d or else - try to do these operations in vectorized form.
* If there is any per component access of these, like `test.x += func(...)` - consider vectorizing  
* Try to keep the complexity of separate parts of the code (like separate functions) as simple as possible keep the code as readible as possible by utilizing code abstractions to its fullest extent, like OOP in Python, or templates and interfaces as well as generics in Slang.
* Do aggressive logic simplifications, try to avoid validity checks for cases which have an extremely low probability of happening, unless this improves debuggability (extremely complex code paths)
* Vector and Matrix math operations are already defined in `spy.math` - prefer using those over recreating helper functions. Import that as `smath`.
* Dont use `\` to split lines in Python if possible.
* Do not add obvious/redundant comments. Method names should be self-documenting. Only comments which describe overall algorithmic logic should be present, unless obvious from the code.
* After a plan has been implemented and approved, commit the changes into the repository with proper description.

# Documentation
* In a separate folder `doc/progress` write down what changes were done after plan has been approved and implemented. These should be written as separate `.md` files with current date and time. 
* The project code should be documented in `.md` files describing the functionality, API as well as algorithms used. Each general part (like `Rendering`) of the project should be in separate files. General description of the project should be in the `README.md` file in the repository root.