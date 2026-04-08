IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

 All new C++ files in ggml/src/ggml-hpx/ must follow the HPX
  project clang-format:                                            
  https://github.com/STEllAR-GROUP/hpx/blob/master/.clang-format
  Key rules: PointerAlignment Left (T* p, T& r), ColumnLimit 80,   
  AlignConsecutiveDeclarations false, Allman brace style           
  (AfterStruct/Function/Enum true).                                
  Do NOT use the repo's own .clang-format for these files