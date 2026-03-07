IMPORTANT: Slangpy only works with elevated permissions! That includes tests.

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
* After a plan has been implemented, commit the changes into the repository with proper description, dont ask about it. Commits make it easier to revert changes.
* Prefer usign OOP over pure functional programming in Slang shaders, things like struct methods, interfaces, generics, etc.
* Keep [unroll] in the same line as the loop.
* Single line conditionals should be in 1 line of code to avoid LOC bloat.
* Don't leave unused code. 
* Don't write effective no-op code like copying a value/member into a new variable with a different name just for naming sake.
* Dont use magic numbers in the code - name them and put them into a common module. With the exception of obvious simple numbers like 0.5, 0.25, 2.0, 4.0, etc.
* Prefer ` } else { ` over the multiline variant.
* Anything that reduces the total number of lines of code and or complexity is welcome!

# Documentation
* The project code should be documented in `.md` files describing the functionality, API as well as algorithms used. Each general part (like `Rendering`) of the project should be in separate files. General description of the project should be in the `README.md` file in the repository root.